import os
import matplotlib.pyplot as plt
import networkx as nx

def parse_nodes(file_path):
    """Parse node information, assuming flexible formatting."""
    nodes = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:  # Format with 6 columns
                node_id, label, prev_nodes, next_nodes, order, node_type = parts
                nodes[node_id] = {
                    "label": label,
                    "prev_nodes": prev_nodes.split(',') if prev_nodes != "NULL" else [],
                    "next_nodes": next_nodes.split(',') if next_nodes != "NULL" else [],
                    "order": int(order) if order.isdigit() else None,  # Safeguard in case 'order' isn't numeric
                    "type": node_type
                }
            elif len(parts) == 5:  # Format with 5 columns
                node_id, label, prev_nodes, order, node_type = parts
                nodes[node_id] = {
                    "label": label,
                    "prev_nodes": prev_nodes.split(',') if prev_nodes != "NULL" else [],
                    "next_nodes": [],
                    "order": int(order) if order.isdigit() else None,  # Safeguard in case 'order' isn't numeric
                    "type": node_type
                }
            else:
                print(f"Warning: Skipping improperly formatted line in {file_path}: {line.strip()}")
    return nodes

def parse_edges(file_path):
    """Parse edge information from the given file."""
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure there are enough parts to unpack
                src, dest, edge_id, edge_type = parts[:4]
                edges.append((src, dest, edge_type))
    return edges

def visualize_contract_graph(contract_name, nodes, edges, output_folder):
    """Generate and save a visualization of the contract graph."""
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for node_id, info in nodes.items():
        label = f"{node_id} ({info['label']})" if 'label' in info else node_id
        G.add_node(node_id, label=label, type=info.get("type", "Normal"))
    
    # Add edges to the graph
    for src, dest, edge_type in edges:
        G.add_edge(src, dest, label=edge_type)
    
    # Set up layout and labels for visualization
    pos = nx.spring_layout(G)
    # Updated to handle cases where 'label' might be missing
    node_labels = {node: G.nodes[node].get("label", node) for node in G.nodes}
    edge_labels = {(src, dest): edge_type for src, dest, edge_type in edges}
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=2000, node_color="lightblue",
            font_size=10, font_weight="bold", arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=8)
    
    # Save the figure
    output_path = os.path.join(output_folder, f"{contract_name}.png")
    plt.title(f"Graph for Contract: {contract_name}")
    plt.savefig(output_path)
    plt.close()
    print(f"Graph saved as {output_path}")

def process_vulnerability_folder(vuln_folder, output_folder, num_contracts):
    """Process a single vulnerability folder to generate contract graphs."""
    sourcecode_folder = os.path.join(vuln_folder, "sourcecode")
    nodes_folder = os.path.join(vuln_folder, "source_graph_data/nodes")
    edges_folder = os.path.join(vuln_folder, "source_graph_data/edges")

    # Get contract file names (limit to `num_contracts`)
    contract_files = sorted(os.listdir(sourcecode_folder))[:num_contracts]

    for contract_file in contract_files:
        contract_name = os.path.splitext(contract_file)[0]
        
        # Paths to nodes and edges files
        nodes_file = os.path.join(nodes_folder, f"{contract_name}.sol")
        edges_file = os.path.join(edges_folder, f"{contract_name}.sol")
        
        # Parse nodes and edges
        if os.path.exists(nodes_file) and os.path.exists(edges_file):
            nodes = parse_nodes(nodes_file)
            edges = parse_edges(edges_file)
            
            # Generate and save the graph visualization
            visualize_contract_graph(contract_name, nodes, edges, output_folder)

def main(data_folder, output_folder, num_contracts_per_vuln):
    """Main function to process all vulnerability folders."""
    vulnerabilities = ["delegatecall", "integeroverflow", "reentrancy", "timestamp"]
    
    for vuln in vulnerabilities:
        vuln_folder = os.path.join(data_folder, vuln)
        vuln_output_folder = os.path.join(output_folder, vuln)
        os.makedirs(vuln_output_folder, exist_ok=True)
        
        print(f"Processing {vuln} contracts...")
        process_vulnerability_folder(vuln_folder, vuln_output_folder, num_contracts_per_vuln)

# User configuration
data_folder = "ESC"  # Path to the main data folder
output_folder = "GraphsOutput"  # Path to the folder where graphs will be saved
num_contracts_per_vuln = 10  # Number of contracts to process per vulnerability folder

os.makedirs(output_folder, exist_ok=True)
main(data_folder, output_folder, num_contracts_per_vuln)
