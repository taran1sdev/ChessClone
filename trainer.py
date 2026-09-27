import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ChessPolicyNet

def train_clone_model(
        dataset, 
        epochs: int = 10, 
        batch_size: int = 32,
        lr: float = 0.001) -> ChessPolicyNet:
    
    model = ChessPolicyNet()
    if len(dataset) == 0:
        print("Dataset is empty.")
        return model

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    print("Beginning model training..")

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
