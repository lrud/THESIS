"""
LSTM model architecture for DVOL forecasting.
"""

import torch
import torch.nn as nn


class DVOLDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTM_DVOL(nn.Module):
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, l2_reg=1e-4):
        super(LSTM_DVOL, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.l2_reg = l2_reg
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take last time step output
        last_output = lstm_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2) ** 2
        return self.l2_reg * l2_loss


def create_model(input_size, hidden_size=128, num_layers=2, dropout=0.3, l2_reg=1e-4, device='cuda'):
    model = LSTM_DVOL(input_size, hidden_size, num_layers, dropout, l2_reg)


def create_model(input_size, hidden_size=128, num_layers=2, dropout=0.3, l2_reg=1e-4, device='cuda'):
    """
    Factory function to create and initialize LSTM model.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        l2_reg: L2 regularization coefficient
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Initialized LSTM model on specified device
    """
    model = LSTM_DVOL(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        l2_reg=l2_reg
    ).to(device)
    
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
