"""
Training utilities for LSTM model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    losses = []
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss = loss + model.l2_regularization()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
    
    return np.mean(losses)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda', 
                early_stop_patience=15, model_save_path='models/lstm_baseline_best.pth'):
    """
    Complete training loop with early stopping.
    
    Returns:
        history: Dictionary with training metrics
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")
    return history
