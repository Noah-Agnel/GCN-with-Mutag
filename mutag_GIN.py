import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

device = torch.device('cpu')

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        def build_mlp():
            return nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
        
        self.lin0 = nn.Linear(in_channels, hidden_channels)
        self.activ0 = nn.ReLU()
        self.conv1 = GINConv(build_mlp())
        self.activ1 = nn.ReLU()
        self.conv2 = GINConv(build_mlp())
        self.activ2 = nn.ReLU()
        self.conv3 = GINConv(build_mlp())
        self.activ3 = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.lin0(x)
        x = self.activ0(x)

        x = self.conv1(x, edge_index)
        x = self.activ1(x)

        x = self.conv2(x, edge_index)
        x = self.activ2(x)

        x = self.conv3(x, edge_index)
        x = self.activ3(x)

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

model = GIN(dataset.num_node_features, hidden_channels=64, out_channels=dataset.num_classes)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
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

max_acc = 0
max_train_acc = 0
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    if train_acc > max_acc:
        max_acc = train_acc
        max_test_acc = test_acc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
