### This script gives tracIn score for all top predicted predicate:27 (treats edge)

import torch
from tqdm import tqdm
import csv
from torch.nn.functional import binary_cross_entropy_with_logits
import argparse
from model import ConvE
from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import (
    JsonLoaderProcessors, Tokenizer, AddToVocab, ConvertTokenToIdx,
    SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState,
    CustomTokenizer, ApplyFunction, ToLower, DictKey2ListMapper,
    StreamToBatch, TargetIdx2MultiTarget
)
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
import numpy as np
import os


def load_vocab(data_path):
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    if os.path.isdir(data_path):
        p = Pipeline(data_path, keys=input_keys)
    else:
        p = Pipeline(os.path.dirname(data_path) if os.path.isfile(data_path) else '.', 
                     keys=input_keys, skip_transformation=True)
    p.load_vocabs()
    return p.state['vocab']


def rebuild_vocab_if_missing(dataset_path):
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    full_path = os.path.join(dataset_path, 'e1rel_to_e2_full.json')
    p = Pipeline(dataset_path, delete_all_previous_data=False, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    d = DatasetStreamer(input_keys)
    d.set_path(full_path)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))
    p.execute(d)
    p.save_vocabs()


def prepare_triple(e1, rel, e2, vocab):
    try:
        e1_idx = vocab['e1'].token2idx[e1.lower()]
        rel_idx = vocab['rel'].token2idx[rel.lower()]
        e2_idx = vocab['e1'].token2idx[e2.lower()]
        return e1_idx, rel_idx, e2_idx
    except KeyError as e:
        raise ValueError(f"Entity or relation not in vocabulary: {str(e)}")


def load_model(checkpoint_path, vocab, args):
    model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to('cuda')
    model.eval()
    return model

def stream_training_triples(file_path, vocab):
        with open(file_path, 'r') as f:
            for line in f:
                e1, rel, e2 = line.strip().split("\t")
                try:
                    e1_idx, rel_idx, e2_idx = prepare_triple(e1, rel, e2, vocab)
                    yield e1_idx, rel_idx, e2_idx
                except ValueError:
                    continue  # skip lines with missing vocab entries

def get_loss_and_gradients(model, indexed_triple, loss_fn, device='cuda'):
    e1_idx, rel_idx, e2_idx = indexed_triple
    e1_tensor = torch.tensor([e1_idx], dtype=torch.long, device=device)
    rel_tensor = torch.tensor([rel_idx], dtype=torch.long, device=device)
    e2_tensor = torch.tensor([e2_idx], dtype=torch.long, device=device)

    model.zero_grad()
    pred = model.predict_score(e1_tensor, rel_tensor, e2_tensor)
    target = torch.tensor([1.0], dtype=torch.float, device=device)
    loss = loss_fn(pred, target)
    loss.backward()

    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    grad_vector = torch.cat(grads)

    return loss.item(), grad_vector



def compute_batch_influence(model, batch_triples, test_grad, loss_fn):
    scores = []
    for train_triple in batch_triples:
        _, train_grad = get_loss_and_gradients(model, train_triple, loss_fn)
        influence = torch.dot(test_grad, train_grad)
        scores.append(influence.item())
    return scores

def compute_influence_scores(model, train_stream, test_triple, loss_fn, batch_size=512):
    
    test_loss, test_grad = get_loss_and_gradients(model, test_triple, loss_fn)
    influence_scores = []

    batch = []
    for train_triple in train_stream:
        batch.append(train_triple)
        if len(batch) == batch_size:
            scores = compute_batch_influence(model, batch, test_grad, loss_fn)
            influence_scores.extend(scores)
            batch = []

    if batch:
        scores = compute_batch_influence(model, batch, test_grad, loss_fn)
        influence_scores.extend(scores)

    return influence_scores, test_loss


def main():
    parser = argparse.ArgumentParser(description='Apply checkpoint model to test triples')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--train-file', type=str, required=True, help='Training triple file')
    parser.add_argument('--test-file', type=str, required=True, help='Test triple file')
    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--embedding-shape1', type=int, default=20)
    parser.add_argument('--hidden-drop', type=float, default=0.3)
    parser.add_argument('--input-drop', type=float, default=0.2)
    parser.add_argument('--feat-drop', type=float, default=0.2)
    parser.add_argument('--use-bias', action='store_true')
    parser.add_argument('--hidden-size', type=int, default=9728)

    args = parser.parse_args()

    # Load vocab and model
    vocab = load_vocab(args.data)
    model = load_model(args.checkpoint, vocab, args)
    loss_fn = binary_cross_entropy_with_logits

    # Load training triples
    # Load all training triples (text + indexed form)
    train_triples_raw = []
    train_triples_indexed = []

    with open(args.train_file, 'r') as tf:
        for line in tf:
            e1_train, rel_train, e2_train = line.strip().split('\t')
            try:
                indexed = prepare_triple(e1_train, rel_train, e2_train, vocab)
                train_triples_indexed.append(indexed)
                train_triples_raw.append((e1_train, rel_train, e2_train))
            except ValueError:
                continue  # skip if any entity/relation is missing from vocab



       # --- Write ALL scores, one CSV per test triple, streaming to disk ---
    import re

    out_dir = os.path.abspath("tracin_for_treatsedge")  # absolute path avoids dir confusion
    os.makedirs(out_dir, exist_ok=True)

    def safe(s, maxlen=80):
        s = s.lower()
        s = re.sub(r'[^a-z0-9._-]+', '_', s)
        return (s[:maxlen] or "x").strip("_")

    def compute_influence_scores_stream(model, train_indexed, test_triple, loss_fn, batch_size=512):
        """Yields (global_train_idx, score) progressively; does NOT accumulate everything in memory."""
        test_loss, test_grad = get_loss_and_gradients(model, test_triple, loss_fn)
        yield ("_TEST_LOSS_", test_loss)

        batch, base = [], 0
        for tr in train_indexed:
            batch.append(tr)
            if len(batch) == batch_size:
                scores = compute_batch_influence(model, batch, test_grad, loss_fn)
                for j, sc in enumerate(scores):
                    yield (base + j, sc)
                base += len(batch)
                batch = []
        if batch:
            scores = compute_batch_influence(model, batch, test_grad, loss_fn)
            for j, sc in enumerate(scores):
                yield (base + j, sc)

    with open(args.test_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="processing test triples"), 1):
            try:
                e1, rel, e2 = line.strip().split("\t")
                test_triple = prepare_triple(e1, rel, e2, vocab)

                # Open ONE per-test CSV; writing happens as we compute batches
                fname = f"t{line_num:05d}__{safe(e1)}__{safe(rel)}__{safe(e2)}.csv"
                path = os.path.join(out_dir, fname)
                print(f"Writing to: {path}")  # makes the path explicit in your logs

                # line-buffered text (buffering=1) helps flush per line on \n
                with open(path, 'w', newline='', buffering=1) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['TestHead','TestRel','TestTail',
                                     'TrainHead','TrainRel','TrainTail','TracInScore'])

                    # stream scores and write as they come
                    gen = compute_influence_scores_stream(
                        model, iter(train_triples_indexed), test_triple, loss_fn, batch_size=512
                    )

                    # first item is a tag carrying test_loss
                    first = next(gen)
                    assert first[0] == "_TEST_LOSS_"
                    test_loss = first[1]
                    print(f"Test Triple {line_num}: {e1}\t{rel}\t{e2}")
                    print(f"Test Loss: {test_loss:.4f}")

                    for idx, score in gen:
                        train_e1, train_rel, train_e2 = train_triples_raw[idx]
                        writer.writerow([e1, rel, e2, train_e1, train_rel, train_e2, float(score)])

                        # flush every 1000 lines for safety/visibility on NFS
                        if idx % 1000 == 0:
                            csvfile.flush()
                            try:
                                os.fsync(csvfile.fileno())
                            except OSError:
                                pass

                # stop after 5 test triples (remove this guard if you want all)
                if line_num >= 5:
                    print("Processed 5 test triples; stopping as requested.")
                    break

            except Exception as e:
                print(f"Skipping line {line_num} due to error: {e}")




if __name__ == '__main__':
    main()
