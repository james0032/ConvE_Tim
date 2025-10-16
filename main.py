import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math

from os.path import join
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler


from evaluation import ranking_and_hits
from model import ConvE, DistMult, Complex

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
import argparse


np.set_printoptions(precision=3)

cudnn.benchmark = True

# Create checkpoints - this is used to save the model state during training; added by DamiOsh
def save_checkpoint(model, optimizer, epoch, loss, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f'model.ckpt-epoch{epoch:04d}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"[Checkpoint] Saved at: {checkpoint_path}")

''' Preprocess knowledge graph using spodernet. '''
def preprocess(data_dir, delete_data=False):
    # full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    # train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    # dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    # test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)
    full_path = os.path.join(data_dir, 'e1rel_to_e2_full.json')
    train_path = os.path.join(data_dir, 'e1rel_to_e2_train.json')
    dev_ranking_path = os.path.join(data_dir, 'e1rel_to_e2_ranking_dev.json')
    test_ranking_path = os.path.join(data_dir, 'e1rel_to_e2_ranking_test.json')


    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))

    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(data_dir, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()


    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
        p.execute(d)


def main(args, model_path):
    dataset_name = os.path.basename(os.path.normpath(args.data))
    if args.preprocess: preprocess(args.data, delete_data=False)
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.data, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

    train_batcher = StreamBatcher(args.data, 'train', args.batch_size, randomize=True, keys=input_keys, loader_threads=args.loader_threads)
    dev_rank_batcher = StreamBatcher(args.data, 'dev_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)
    test_rank_batcher = StreamBatcher(args.data, 'test_ranking', args.test_batch_size, randomize=False, loader_threads=args.loader_threads, keys=input_keys)


    if args.model is None:
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'conve':
        model = ConvE(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'distmult':
        model = DistMult(args, vocab['e1'].num_token, vocab['rel'].num_token)
    elif args.model == 'complex':
        model = Complex(args, vocab['e1'].num_token, vocab['rel'].num_token)
    else:
        log.info('Unknown model: {0}', args.model)
        raise Exception("Unknown model!")

    train_batcher.at_batch_prepared_observers.insert(1,TargetIdx2MultiTarget(num_entities, 'e2_multi1', 'e2_multi1_binary'))


    eta = ETAHook('train', print_every_x_batches=args.log_interval)
    train_batcher.subscribe_to_events(eta)
    train_batcher.subscribe_to_start_of_epoch_event(eta)
    train_batcher.subscribe_to_events(LossHook('train', print_every_x_batches=args.log_interval))

    #model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.resume:
        model_params = torch.load(model_path)
        print(model)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
            print(key, size, count)
        print(np.array(total_param_size).sum())
        model.load_state_dict(model_params)
        model.eval()
        ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')
        ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
    else:
        model.init()

    total_param_size = []
    params = [value.numel() for value in model.parameters()]
    print(params)
    print(np.sum(params))

    ##### Using amp for mixed precision training #####
    # --- AMP Implementation Start ---
    if args.use_amp:
        if device.type == 'cuda':
            print("Using automatic mixed precision (AMP) for training.")
            scaler = GradScaler()
        else:
            print("AMP is enabled but a CUDA device is not available. Running in FP32.")
            scaler = None
    else:
        scaler = None

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.l2)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        loss_fn = torch.nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for binary classification for autograd
        for batch_idx, batch_data in enumerate(train_batcher):
            e1 = batch_data['e1'].to(device)
            rel = batch_data['rel'].to(device)
            e2_multi1_binary = batch_data['e2_multi1_binary'].to(device).float()
            
            # Add label smoothing (missing in Code 1)
            e2_multi1_binary = ((1.0-args.label_smoothing)*e2_multi1_binary) + (1.0/e2_multi1_binary.size(1))

            opt.zero_grad()

            with autocast(enabled=args.use_amp and device.type == 'cuda'):
                pred = model.forward(e1, rel)
                loss = loss_fn(pred, e2_multi1_binary)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

                # Clear cache every 100 batches
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

            # Add loss tracking (missing in Code 1)
            total_loss += loss.item()
            train_batcher.state.loss = loss.cpu()

    #### End of amp addition #####


   
    #### Original ConvE training loop - commented out and replaced by the above AMP version ####
    # opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    # for epoch in range(args.epochs):
    #     model.train()
    #     total_loss = 0.0  # Initialize total loss for the epoch. Added by DamiOsh
    #     for i, str2var in enumerate(train_batcher):
    #         opt.zero_grad()
    #         e1 = str2var['e1']
    #         rel = str2var['rel']
    #         e2_multi = str2var['e2_multi1_binary'].float()
    #         # label smoothing
    #         e2_multi = ((1.0-args.label_smoothing)*e2_multi) + (1.0/e2_multi.size(1))

    #         pred = model.forward(e1, rel)
    #         loss = model.loss(pred, e2_multi)
    #         loss.backward()
    #         opt.step()

    #         train_batcher.state.loss = loss.cpu()

# Save the checkpoint here: added by DamiOsh
        #total_loss += loss.item()  # Accumulate loss for the checkpoint. Added by DamiOsh
        checkpoint_dir = os.path.join(args.data, 'checkpoints')
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, opt, epoch, total_loss, model_dir=checkpoint_dir)

        print('saving to {0}'.format(model_path))
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            if epoch % 5 == 0 and epoch > 0:
                ranking_and_hits(model, dev_rank_batcher, vocab, 'dev_evaluation')
            if epoch % 5 == 0:
                if epoch > 0:
                    ranking_and_hits(model, test_rank_batcher, vocab, 'test_evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link prediction for knowledge graphs')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision (AMP) training.')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing/validation (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='FB15k-237', help='Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}, default: FB15k-237')
    parser.add_argument('--l2', type=float, default=0.0, help='Weight decay value to use in the optimizer. Default: 0.0')
    parser.add_argument('--model', type=str, default='conve', help='Choose from: {conve, distmult, complex}')
    parser.add_argument('--embedding-dim', type=int, default=200, help='The embedding dimension (1D). Default: 200')
    parser.add_argument('--embedding-shape1', type=int, default=20, help='The first dimension of the reshaped 2D embedding. The second dimension is infered. Default: 20')
    parser.add_argument('--hidden-drop', type=float, default=0.3, help='Dropout for the hidden layer. Default: 0.3.')
    parser.add_argument('--input-drop', type=float, default=0.2, help='Dropout for the input embeddings. Default: 0.2.')
    parser.add_argument('--feat-drop', type=float, default=0.2, help='Dropout for the convolutional features. Default: 0.2.')
    parser.add_argument('--lr-decay', type=float, default=0.995, help='Decay the learning rate by this factor every epoch. Default: 0.995')
    parser.add_argument('--loader-threads', type=int, default=4, help='How many loader threads to use for the batch loaders. Default: 4')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the dataset. Needs to be executed only once. Default: 4')
    parser.add_argument('--resume', action='store_true', help='Resume a model.')
    parser.add_argument('--use-bias', action='store_true', help='Use a bias in the convolutional layer. Default: True')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value to use. Default: 0.1')
    parser.add_argument('--hidden-size', type=int, default=9728, help='The side of the hidden layer. The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).')

    args = parser.parse_args()



    # parse console parameters and set global variables
    Config.backend = 'pytorch'
    Config.cuda = True
    Config.embedding_dim = args.embedding_dim
    #Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG


    model_name = '{2}_{0}_{1}'.format(args.input_drop, args.hidden_drop, args.model)
    #model_path = 'saved_models/{0}_{1}.model'.format(args.data, model_name)
    model_path = os.path.join(args.data, f"{model_name}.model")

    torch.manual_seed(args.seed)
    main(args, model_path)
