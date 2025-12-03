import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, GlobalAttention, GraphNorm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
import random
from torch_geometric.nn import AttentionalAggregation
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split


# Initialize label_to_index, type_to_index, and label_embeddings
label_to_index = {}
type_to_index = {}
embedding_dim = 16
label_embeddings = None  # To be initialized once label_to_index is populated
type_embeddings = None  # To be initialized once type_to_index is populated


# Set seeds for reproducibility
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Weighted Cross-Entropy Loss for Improved Balance
class WeightedCrossEntropy(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropy, self).__init__()
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.class_weights)


# Compute class weights
def compute_class_weights(labels):
    class_counts = Counter(labels)
    total_count = sum(class_counts.values())
    weights = {label: total_count / (len(class_counts) * count) for label, count in class_counts.items()}
    return torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float)


# Stratified data loader
def stratified_data_split(data, labels, train_ratio=0.7, val_ratio=0.15):
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, stratify=labels, test_size=1 - train_ratio, random_state=42
    )
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, stratify=temp_labels, test_size=1 - val_ratio_adjusted, random_state=42
    )
    return train_data, val_data, test_data

# Balanced data loader
def balanced_data_loader(data, batch_size=16):
    labels = [d.y.item() for d in data]
    class_weights = compute_class_weights(labels)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(data, batch_size=batch_size, sampler=sampler)


# Node Parsing
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
                    "order": int(order),
                    "type": node_type
                }
            elif len(parts) == 5:  # Format with 5 columns
                node_id, label, prev_nodes, order, node_type = parts
                nodes[node_id] = {
                    "label": label,
                    "prev_nodes": prev_nodes.split(',') if prev_nodes != "NULL" else [],
                    "next_nodes": [],
                    "order": int(order),
                    "type": node_type
                }
            else:
                print(f"Warning: Skipping improperly formatted line in {file_path}: {line.strip()}")
    return nodes


# Edge Parsing
def parse_edges(file_path):
    edges = []
    edge_types = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                print(f"Warning: Skipping improperly formatted line in {file_path}: {line.strip()}")
                continue
            src, dest, edge_id, edge_type = parts
            edges.append((src, dest))
            edge_types.append(edge_type)
    return edges, edge_types


# Label Mapping
def get_label_index(label):
    if label not in label_to_index:
        label_to_index[label] = len(label_to_index)
    return label_to_index[label]


def get_type_index(node_type):
    if node_type not in type_to_index:
        type_to_index[node_type] = len(type_to_index)
    return type_to_index[node_type]


# Load Graph Data
def load_graph_data(folder, label):
    graphs = []
    sourcecode_folder = os.path.join(folder, "sourcecode")
    nodes_folder = os.path.join(folder, "source_graph_data/nodes")
    edges_folder = os.path.join(folder, "source_graph_data/edges")

    contract_files = [f for f in os.listdir(sourcecode_folder) if f.endswith('.sol')]

    for file_name in contract_files:
        contract_name = os.path.splitext(file_name)[0]
        nodes_file = os.path.join(nodes_folder, f"{contract_name}.sol")
        edges_file = os.path.join(edges_folder, f"{contract_name}.sol")

        if os.path.exists(nodes_file) and os.path.exists(edges_file):
            nodes = parse_nodes(nodes_file)
            edges, edge_types = parse_edges(edges_file)

            if not edges:
                print(f"Warning: No edges data for contract {contract_name}")
                continue

            node_indices = {node_id: idx for idx, node_id in enumerate(nodes)}
            x = torch.tensor([[get_label_index(nodes[node_id]['label']), get_type_index(nodes[node_id]['type']), nodes[node_id]['order']]
                              for node_id in nodes], dtype=torch.float)

            valid_edges = [(node_indices[src], node_indices[dest]) for src, dest in edges if src in node_indices and dest in node_indices]
            if not valid_edges:
                print(f"Warning: No valid edges after filtering for contract {contract_name}")
                continue
            
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([get_type_index(edge_type) for edge_type in edge_types], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))
            graphs.append(data)
        else:
            print(f"Warning: Missing nodes or edges file for {contract_name}")

    return graphs


# Positional Encoding for Graph Nodes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:  # Handle even dimensions
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"seq_len ({seq_len}) exceeds max_len ({self.max_len}) in PositionalEncoding.")
        return x + self.pe[:, :seq_len, :].to(x.device)


# Model Definition
class EnhancedGATLSTMWithAttention(nn.Module):
    def __init__(self, gat_hidden_dim=128, lstm_hidden_dim=256, output_dim=4, n_heads=8, embedding_dim=16, dropout_rate=0.3):
        super(EnhancedGATLSTMWithAttention, self).__init__()
        self.node_embedding = nn.Embedding(len(label_to_index), embedding_dim)
        self.type_embedding = nn.Embedding(len(type_to_index), embedding_dim)
        self.input_feature_dim = embedding_dim * 2 + 1
        self.positional_encoding = PositionalEncoding(self.input_feature_dim)
        self.input_proj = nn.Linear(self.input_feature_dim, gat_hidden_dim * n_heads)
        self.gat1 = GATv2Conv(gat_hidden_dim * n_heads, gat_hidden_dim, heads=n_heads, concat=True)
        self.gat1_norm = GraphNorm(gat_hidden_dim * n_heads)
        self.gat1_dropout = nn.Dropout(dropout_rate)
        self.gat2 = GATv2Conv(gat_hidden_dim * n_heads, gat_hidden_dim, heads=n_heads, concat=True)
        self.gat2_norm = GraphNorm(gat_hidden_dim * n_heads)
        self.gat2_dropout = nn.Dropout(dropout_rate)
        self.residual_connection = nn.Linear(gat_hidden_dim * n_heads, gat_hidden_dim * n_heads)
        self.lstm = nn.LSTM(input_size=gat_hidden_dim * n_heads, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.lstm_norm = nn.LayerNorm(lstm_hidden_dim)
        self.attention = nn.Linear(lstm_hidden_dim, 1)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        label_emb = self.node_embedding(x[:, 0].long())
        type_emb = self.type_embedding(x[:, 1].long())
        x = torch.cat([label_emb, type_emb, x[:, 2:3]], dim=1)
        x = self.positional_encoding(x.unsqueeze(1)).squeeze(1)
        x_proj = self.input_proj(x)
        x1 = F.elu(self.gat1_norm(self.gat1(x_proj, edge_index)))
        x1 = self.gat1_dropout(x1)
        x2 = F.elu(self.gat2_norm(self.gat2(x1, edge_index)) + self.residual_connection(x1))
        x2 = self.gat2_dropout(x2)
        x = x2.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.lstm_norm(x[:, -1, :])
        x = global_mean_pool(x, batch)  # Pool outputs to match batch size
        x = self.fc_dropout(x)
        return F.log_softmax(self.fc(x), dim=1)



# Training Function
def train(model, loader, optimizer, criterion, clip_value=1.0):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        #print(f"Batch size: {data.batch.size(0)}, Target size: {data.y.size(0)}")  # Debugging line
        output = model(data)
        #print(f"Output size: {output.size(0)}")  # Debugging line
        assert output.size(0) == data.y.size(0), \
            f"Mismatch: Output batch size {output.size(0)} != Target batch size {data.y.size(0)}"
        loss = criterion(output, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



# Validation Function
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            pred = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(data.y.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), accuracy

# Testing and Evaluation Function
def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            output = model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(data.y.cpu().numpy())
    return all_labels, all_preds

def evaluate_model(labels, predictions):
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average="weighted", zero_division=0)
    rec = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

# Main Function
def main(data_folder, epochs=100, initial_lr=0.001, gat_hidden_dim=128, lstm_hidden_dim=256, patience=5):
    global label_to_index, type_to_index, label_embeddings, type_embeddings

    # Define vulnerabilities and assign labels
    labels = {"delegatecall": 0, "integeroverflow": 1, "reentrancy": 2, "timestamp": 3}
    all_data = []

    # Load and shuffle data
    for vuln, label in labels.items():
        vuln_folder = os.path.join(data_folder, vuln)
        graphs = load_graph_data(vuln_folder, label)
        all_data.extend(graphs)

    # Extract labels for stratified splitting
    all_labels = [data.y.item() for data in all_data]

    # Stratified data splitting
    train_data, val_data, test_data = stratified_data_split(all_data, all_labels)

    # Compute class weights
    class_weights = compute_class_weights(all_labels)

    # Create data loaders
    train_loader = balanced_data_loader(train_data, batch_size=16)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Initialize model
    model = EnhancedGATLSTMWithAttention(
        gat_hidden_dim=gat_hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        output_dim=len(labels),
        n_heads=8,
        dropout_rate=0.3
    )

    # Define optimizer, scheduler, and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = WeightedCrossEntropy(class_weights=class_weights.to(model.fc.weight.device))

    # Training loop with early stopping
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        # Step the learning rate scheduler
        scheduler.step()

        # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if epochs_no_improve >= patience:
            print("Early stopping due to no improvement in validation accuracy.")
            break

    # Final testing and evaluation
    labels, predictions = test(model, test_loader)
    evaluate_model(labels, predictions)


# Configuration
data_folder = "ESC"
epochs = 100
main(data_folder, epochs=epochs, initial_lr=0.001, gat_hidden_dim=128, lstm_hidden_dim=256)

