import os
import torch
import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.data import Data

# Define label mappings
LABELS = {"delegatecall": 0, "integeroverflow": 1, "reentrancy": 2, "timestamp": 3}
DATA_FOLDER = "ESC"  # Specify the main data folder

# Function to parse labels from the data files
def parse_labels(data_folder):
    all_data = []
    for vuln_name, label_id in LABELS.items():
        vuln_folder = os.path.join(data_folder, vuln_name, "source_graph_data", "nodes")
        if not os.path.exists(vuln_folder):
            print(f"Warning: Folder for {vuln_name} not found.")
            continue
        for file_name in os.listdir(vuln_folder):
            if file_name.endswith('.sol'):
                # Create dummy data points with labels for counting
                data = Data(y=torch.tensor([label_id]))
                all_data.append(data)
    return all_data

# Count the occurrences of each vulnerability type in the dataset
def count_vulnerabilities(data):
    labels = [d.y.item() for d in data]  # Extract labels from each data entry
    counts = Counter(labels)  # Count occurrences of each label
    return counts

# Display summary of the dataset
def display_dataset_summary(data, counts):
    total_samples = len(data)
    unique_labels = len(counts)
    print("Dataset Summary")
    print("------------------")
    print(f"Total Samples: {total_samples}")
    print(f"Unique Labels: {unique_labels}")
    print("Samples per Vulnerability Type:")
    for vuln, idx in LABELS.items():
        print(f"{vuln.capitalize()}: {counts[idx]}")

# Plot the bar graph of vulnerability counts
def plot_vulnerability_distribution(counts):
    # Prepare data for plotting
    labels = [vuln for vuln, idx in LABELS.items()]  # Vulnerability names
    values = [counts[idx] for idx in range(len(LABELS))]  # Count for each label index

    # Plot bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Vulnerability Type')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Vulnerability Types in Dataset')
    plt.xticks(rotation=45)
    plt.show()

# Main execution
if __name__ == "__main__":
    all_data = parse_labels(DATA_FOLDER)  # Load and label data
    counts = count_vulnerabilities(all_data)  # Count each vulnerability type
    display_dataset_summary(all_data, counts)  # Display dataset summary
    plot_vulnerability_distribution(counts)  # Plot the results
