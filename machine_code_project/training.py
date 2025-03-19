# -*- coding: utf-8 -*-
import time, math, torch, torch.nn as nn, torch.optim as optim
from datasets import collate_fn_text
from model import TransformerLM, VOCAB_SIZE, PAD_IDX
from torch.utils.data import DataLoader, random_split, ConcatDataset

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs).view(-1, model(inputs).size(-1))
            loss = criterion(logits, targets.view(-1))
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0)
    avg_loss = total_loss / total_tokens if total_tokens else float('inf')
    return avg_loss, math.exp(avg_loss)

def train_model_standard(model, train_loader, val_loader, num_epochs, lr, device,
                         clip_grad=1.0, checkpoint_path="transformer_model.pt", log_fn=print, 
                         progress_callback=None, pause_callback=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    for epoch in range(1, num_epochs+1):
        if pause_callback:
            pause_callback()
        epoch_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs).view(-1, model(inputs).size(-1))
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        log_fn(f"Epoch {epoch}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
        if progress_callback:
            progress_callback(epoch, val_loss)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)
    return model

def train_model_diffusion(model, train_loader, val_loader, num_epochs, lr, device,
                          clip_grad=1.0, checkpoint_path="transformer_model_diffusion.pt", log_fn=print, 
                          progress_callback=None, pause_callback=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    for epoch in range(1, num_epochs+1):
        if pause_callback:
            pause_callback()
        epoch_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs).view(-1, model(inputs).size(-1))
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        log_fn(f"Epoch {epoch}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
        if progress_callback:
            progress_callback(epoch, val_loss)
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)
    return model

# Функции train_model_diffusion и train_model_adaptive можно реализовать аналогично

# Класс TrainingWorker, наследник QThread, реализуется в GUI-модуле (см. gui/gui_main.py)
