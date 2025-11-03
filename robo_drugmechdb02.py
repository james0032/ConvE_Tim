#######------- This script is to filter out all treats edges from any resource in robokop and extract 
#mechanistic paths from drugmechdb only.--------######
import gzip
import json
from collections import defaultdict
import torch
from torch_geometric.data import Data
import networkx as nx

def extract_treats_edges(edges_file='roboedges.jsonl.gz'):
    """Extract all edges with 'biolink:treats' predicate"""
    
    treats_edges = []
    total_lines = 0
    
    print("Step 1: Extracting 'biolink:treats' edges...")
    with gzip.open(edges_file, 'rt') as f:
        for line in f:
            total_lines += 1
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} lines... Found {len(treats_edges)} treats edges")
            
            record = json.loads(line)
            if record.get('predicate') == 'biolink:treats':
                treats_edges.append({
                    'subject': record['subject'],
                    'object': record['object'],
                    'source': record.get('primary_knowledge_source', 'unknown')
                })
    
    print(f"  Found {len(treats_edges)} 'biolink:treats' edges")
    return treats_edges

def build_drugmechdb_graph(edges_file='roboedges.jsonl.gz'):
    """Build graph from DrugMechDB edges only"""
    
    edges = []
    node_to_idx = {}
    node_idx = 0
    edge_attributes = []
    
    print("\nStep 2: Building DrugMechDB graph...")
    with gzip.open(edges_file, 'rt') as f:
        for line in f:
            record = json.loads(line)
            
            if record.get('primary_knowledge_source') == 'infores:drugmechdb':
                subj = record['subject']
                obj = record['object']
                pred = record.get('predicate', 'unknown')
                
                if subj not in node_to_idx:
                    node_to_idx[subj] = node_idx
                    node_idx += 1
                if obj not in node_to_idx:
                    node_to_idx[obj] = node_idx
                    node_idx += 1
                
                edges.append([node_to_idx[subj], node_to_idx[obj]])
                edge_attributes.append({
                    'predicate': pred,
                    'subject_id': subj,
                    'object_id': obj
                })
    
    print(f"  DrugMechDB graph: {len(node_to_idx)} nodes, {len(edges)} edges")
    
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return edge_index, idx_to_node, edge_attributes, node_to_idx

def build_edge_lookup(edge_index, edge_attributes):
    """Build lookup for edge attributes"""
    edge_lookup = {}
    edges = edge_index.t().numpy()
    
    for i, (src, dst) in enumerate(edges):
        edge_lookup[(src, dst)] = edge_attributes[i]
    
    return edge_lookup

def find_mechanistic_paths_for_treats_edge(treats_edge, G, idx_to_node, node_to_idx, edge_lookup, max_length=5):
    """Find all DrugMechDB paths explaining a single treats edge"""
    
    drug = treats_edge['subject']
    disease = treats_edge['object']
    
    # Check if both nodes exist in DrugMechDB graph
    if drug not in node_to_idx or disease not in node_to_idx:
        return []
    
    drug_idx = node_to_idx[drug]
    disease_idx = node_to_idx[disease]
    
    # Find all simple paths from drug to disease
    paths = []
    try:
        nx_paths = nx.all_simple_paths(G, drug_idx, disease_idx, cutoff=max_length)
        
        for path in nx_paths:
            # Build path with edge attributes
            path_with_attrs = []
            for j in range(len(path) - 1):
                src_idx = path[j]
                dst_idx = path[j + 1]
                
                edge_attr = edge_lookup.get((src_idx, dst_idx), {
                    'predicate': 'unknown',
                    'subject_id': idx_to_node[src_idx],
                    'object_id': idx_to_node[dst_idx]
                })
                
                path_with_attrs.append({
                    'source_node': idx_to_node[src_idx],
                    'target_node': idx_to_node[dst_idx],
                    'predicate': edge_attr['predicate']
                })
            
            paths.append({
                'length': len(path),
                'nodes': [idx_to_node[node] for node in path],
                'edges': path_with_attrs
            })
    except nx.NetworkXNoPath:
        pass
    
    return paths

def find_all_mechanistic_paths(treats_edges, edge_index, idx_to_node, node_to_idx, edge_attributes, max_length=5):
    """Find DrugMechDB mechanistic paths for all treats edges"""
    
    print("\nStep 3: Finding DrugMechDB mechanistic paths for each treats edge...")
    
    # Build NetworkX graph and edge lookup
    G = nx.DiGraph()
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    edge_lookup = build_edge_lookup(edge_index, edge_attributes)
    
    print(f"  DrugMechDB graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # For each treats edge, find mechanistic paths
    results = []
    treats_with_paths = 0
    total_paths_found = 0
    
    for i, treats_edge in enumerate(treats_edges):
        if i % 10 == 0:
            print(f"  Processing treats edge {i}/{len(treats_edges)}...")
        
        paths = find_mechanistic_paths_for_treats_edge(
            treats_edge, G, idx_to_node, node_to_idx, edge_lookup, max_length
        )
        
        if paths:
            treats_with_paths += 1
            total_paths_found += len(paths)
        
        results.append({
            'treats_edge': treats_edge,
            'num_mechanistic_paths': len(paths),
            'mechanistic_paths': paths
        })
    
    print(f"\n  Treats edges with mechanistic paths: {treats_with_paths}/{len(treats_edges)}")
    print(f"  Total mechanistic paths found: {total_paths_found}")
    
    return results

def summarize_results(results):
    """Summarize the mechanistic paths found"""
    
    print(f"\n{'='*70}")
    print("MECHANISTIC PATH SUMMARY")
    print(f"{'='*70}")
    
    # Overall statistics
    total_treats = len(results)
    treats_with_paths = sum(1 for r in results if r['num_mechanistic_paths'] > 0)
    total_paths = sum(r['num_mechanistic_paths'] for r in results)
    
    print(f"Total treats edges: {total_treats}")
    print(f"Treats edges with DrugMechDB paths: {treats_with_paths}")
    print(f"Treats edges WITHOUT DrugMechDB paths: {total_treats - treats_with_paths}")
    print(f"Total mechanistic paths: {total_paths}")
    
    if treats_with_paths > 0:
        avg_paths = total_paths / treats_with_paths
        print(f"Average paths per treats edge (when paths exist): {avg_paths:.2f}")
    
    # Path length distribution
    all_paths = []
    for r in results:
        all_paths.extend(r['mechanistic_paths'])
    
    if all_paths:
        lengths = [p['length'] for p in all_paths]
        length_dist = defaultdict(int)
        for l in lengths:
            length_dist[l] += 1
        
        print(f"\nMechanistic path length distribution:")
        for length in sorted(length_dist.keys()):
            print(f"  Length {length}: {length_dist[length]} paths")
        
        # Predicate patterns
        predicate_sequences = []
        for path in all_paths:
            pred_seq = tuple(edge['predicate'] for edge in path['edges'])
            predicate_sequences.append(pred_seq)
        
        pred_counts = defaultdict(int)
        for seq in predicate_sequences:
            pred_counts[seq] += 1
        
        print(f"\nTop 10 predicate patterns in mechanistic paths:")
        for seq, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            pred_str = ' → '.join(seq)
            print(f"  {pred_str}: {count} paths")
    
    # Show examples
    print(f"\nSample treats edges with their mechanistic paths:")
    examples_shown = 0
    for r in results:
        if r['num_mechanistic_paths'] > 0 and examples_shown < 5:
            print(f"\n  Treats edge:")
            print(f"    {r['treats_edge']['subject']} --[biolink:treats]--> {r['treats_edge']['object']}")
            print(f"    Source: {r['treats_edge']['source']}")
            print(f"    DrugMechDB mechanistic paths found: {r['num_mechanistic_paths']}")
            
            # Show first mechanistic path
            if r['mechanistic_paths']:
                path = r['mechanistic_paths'][0]
                print(f"    Example mechanistic path (length {path['length']}):")
                for edge in path['edges']:
                    print(f"      {edge['source_node']} --[{edge['predicate']}]--> {edge['target_node']}")
            
            examples_shown += 1

def save_results(results, output_file='treats_mechanistic_paths.json'):
    """Save results to JSON file"""
    
    print(f"\n{'='*70}")
    print(f"Saving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'total_treats_edges': len(results),
            'treats_edges_with_paths': sum(1 for r in results if r['num_mechanistic_paths'] > 0),
            'total_mechanistic_paths': sum(r['num_mechanistic_paths'] for r in results),
            'results': results
        }, f, indent=2)
    
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved to {output_file} ({file_size_mb:.2f} MB)")

def save_results_txt(results, output_txt='treats_mechanistic_paths.txt'):
    """Save mechanistic paths to a .txt file in the format:
       Drug, Disease, [list of intermediate nodes in the mechanistic path]
    """
    print(f"\n{'='*70}")
    print(f"Saving simplified results to {output_txt}...")

    with open(output_txt, 'w') as f:
        f.write("Drug, Disease, [Intermediate Nodes]\n")
        for r in results:
            drug = r['treats_edge']['subject']
            disease = r['treats_edge']['object']
            for path in r['mechanistic_paths']:
                nodes = path['nodes']
                if len(nodes) > 2:
                    intermediates = nodes[1:-1]  # exclude drug and disease
                else:
                    intermediates = []
                intermediates_str = "[" + ", ".join(intermediates) + "]"
                f.write(f"{drug}, {disease}, {intermediates_str}\n")

    import os
    file_size_kb = os.path.getsize(output_txt) / 1024
    print(f"✓ Saved to {output_txt} ({file_size_kb:.2f} KB)")

def main():
    # Step 1: Extract all treats edges
    treats_edges = extract_treats_edges()
    
    # Step 2: Build DrugMechDB graph
    edge_index, idx_to_node, edge_attributes, node_to_idx = build_drugmechdb_graph()
    
    # Step 3: Find mechanistic paths for each treats edge
    results = find_all_mechanistic_paths(
        treats_edges, edge_index, idx_to_node, node_to_idx, edge_attributes, max_length=5
    )
    
    # Step 4: Summarize results
    summarize_results(results)
    
    # Step 5: Save results
    save_results(results)
    save_results_txt(results)
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    
    return results

if __name__ == "__main__":
    results = main()