import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

def parse_biomedical_data_debug(file_path):
    """Parse function that stores predicate information"""
    nodes = set()
    edges = []
    edge_types = []
    triple_lines = []
    edge_predicates = []  # NEW: Store original predicate strings
    edge_triples = []     # NEW: Store complete triples for output
    
    with open(file_path, 'r') as f:
        for line in f:
            triple_lines.append(line.strip())
            parts = line.strip().split()
            if len(parts) >= 3:
                source = parts[0]
                predicate = parts[1]
                target = parts[2]
                
                nodes.add(source)
                nodes.add(target)
                edges.append((source, target))
                edge_predicates.append(predicate)  # NEW: Store predicate
                edge_triples.append((source, predicate, target))  # NEW: Store full triple
                
                if 'predicate:' in predicate:
                    edge_type = int(predicate.split(':')[1])
                    edge_types.append(edge_type)
    
    # Create node mapping
    node_list = list(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Convert edges to tensor format
    edge_sources = []
    edge_targets = []
    edge_info = []  # Store original edge info for debugging
    
    for src, tgt in edges:
        edge_sources.append(node_to_idx[src])
        edge_targets.append(node_to_idx[tgt])
        edge_info.append(f"{src} -> {tgt}")
    
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    
    # Create node features
    x = torch.arange(len(node_list)).unsqueeze(-1).float()
    
    # Create data object
    data = Data(
        x=x,
        edge_index=edge_index,
        node_names=node_list,
        edge_types=edge_types if edge_types else None,
        triple_lines=triple_lines,
        edge_info=edge_info,
        edge_predicates=edge_predicates,  # NEW: Store predicates
        edge_triples=edge_triples         # NEW: Store full triples
    )
    
    return data, node_to_idx

def filter_graph_by_edge_hops(data, query_head, query_tail, k):
    """Extract k-hop subgraph and return triples with predicates"""
    
    # Find node indices for query edge
    head_idx = None
    tail_idx = None
    for idx, name in enumerate(data.node_names):
        if name == query_head:
            head_idx = idx
        if name == query_tail:
            tail_idx = idx
    
    if head_idx is None or tail_idx is None:
        print(f"Query edge nodes not found: {query_head} -> {query_tail}")
        return None
    
    print(f"\n=== EXTRACTING {k}-HOP SUBGRAPH AROUND EDGE: {query_head} -> {query_tail} ===")
    
    # Start from BOTH nodes of the query edge
    start_nodes = [head_idx, tail_idx]
    
    # Get k-hop subgraph around both nodes
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=start_nodes,
        num_hops=k,
        edge_index=data.edge_index,
        relabel_nodes=False
    )
    
    # NEW: Extract the triples with predicates using the edge_mask
    filtered_triples = []
    for i, (included, triple) in enumerate(zip(edge_mask, data.edge_triples)):
        if included:
            filtered_triples.append(triple)
    
    print(f"subset node names: {[data.node_names[i] for i in subset.tolist()]}")
    print(f"Number of edges in subgraph: {edge_index.shape[1]}")
    print(f"Number of triples in subgraph: {len(filtered_triples)}")
    
    # Create subgraph
    subgraph = data.__class__()
    subgraph.x = data.x[subset]
    subgraph.edge_index = edge_index
    
    # Preserve node names
    if hasattr(data, 'node_names'):
        subgraph.node_names = [data.node_names[i] for i in subset.tolist()]
    
    # NEW: Preserve predicates and triples
    if hasattr(data, 'edge_predicates'):
        subgraph.edge_predicates = [data.edge_predicates[i] for i in range(len(data.edge_predicates)) if edge_mask[i]]
    
    if hasattr(data, 'edge_triples'):
        subgraph.edge_triples = filtered_triples
    
    return subgraph, subset, mapping, edge_mask, filtered_triples  # NEW: Return filtered_triples

def get_k_hop_triples(data, query_head, query_tail, k):
    """MAIN FUNCTION: Get k-hop connected triples with predicates in original format"""
    
    # Extract the subgraph
    result = filter_graph_by_edge_hops(data, query_head, query_tail, k)
    if result is None:
        return []
    
    subgraph, subset, mapping, edge_mask, filtered_triples = result
    
    print(f"\n{'='*60}")
    print(f"K-HOP TRIPLES FOR EDGE: {query_head} -> {query_tail} (k={k})")
    print(f"{'='*60}")
    
    # Return triples in the original format
    return filtered_triples

# # Create test data file with connected graph
# test_data = """CHEBI:16610\tpredicate:18\tNCBIGene:5358
# NCBIGene:23530\tpredicate:8\tNCBIGene:64943
# MONDO:0030517\tpredicate:13\tHP:0003487
# NCBIGene:943\tpredicate:7\tNCBIGene:7185
# NCBIGene:5358\tpredicate:5\tMONDO:0030517
# HP:0003487\tpredicate:3\tCHEBI:16610
# NCBIGene:64943\tpredicate:2\tNCBIGene:943"""

# # Write test data to file
# with open('test_biomedical.txt', 'w') as f:
#     f.write(test_data)

# Run the debug
print("=== PARSING ORIGINAL DATA ===")
data, node_to_idx = parse_biomedical_data_debug('/proj/jchunglab/projects/ec_moa/ConvE_2DCNN/ConvE_subgraph/ConvE/data/ROBOKOP_30fd_baseline2_CCGGDD_noSubclassOf_from_ConvE_benchmark/train.txt')

print(f"\n=== FULL GRAPH INFO ===")
print(f"Total nodes: {len(data.node_names)}")
print(f"Total edges: {data.edge_index.shape[1]}")
print(f"All triples in original graph:")
for triple in data.edge_triples:
    print(f"  {triple[0]}\t{triple[1]}\t{triple[2]}")

# Test: Get k-hop triples around the query edge
query_head = "UNII:U59UGK3IPC"
query_tail = "MONDO:0005314"

print(f"\n{'='*60}")
print(f"QUERY EDGE: {query_head} -> {query_tail}")
print(f"{'='*60}")

# # Get 1-hop triples
# triples_1hop = get_k_hop_triples(data, query_head, query_tail, k=1)
# print(f"\n1-HOP TRIPLES ({len(triples_1hop)} triples):")
# for triple in triples_1hop:
#     print(f"{triple[0]}\t{triple[1]}\t{triple[2]}")

# Get 2-hop triples  
triples_2hop = get_k_hop_triples(data, query_head, query_tail, k=2)
print(f"\n2-HOP TRIPLES ({len(triples_2hop)} triples):")
# Save to .txt file
output_filename = f"{query_head}_{query_tail}_khop2_triples.txt"
with open(output_filename, 'w') as f:
    for triple in triples_2hop:
        triple_line = f"{triple[0]}\t{triple[1]}\t{triple[2]}\n"
        print(triple_line.strip())  # Print to console
        f.write(triple_line)        # Write to file

print(f"\nTriples saved to: {output_filename}")





