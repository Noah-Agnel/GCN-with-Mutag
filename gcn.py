import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

device = torch.device('cpu')

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.activ1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.activ2 = nn.ReLU()
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout()
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.activ1(x)
        x = self.conv2(x, edge_index)
        x = self.activ2(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.lin(x) # Classify graph embeddings
        return x

split_index = int(len(dataset) * 0.8)
dataset = dataset.shuffle()
train_dataset = dataset[:split_index]
test_dataset = dataset[split_index:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = GCN(dataset.num_node_features, hidden_channels=64, out_channels=dataset.num_classes)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

model = model.to(device)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # forward pass
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
