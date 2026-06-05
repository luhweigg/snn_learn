import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes=10):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for events, targets in dataloader:
        events, targets = events.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(events)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device, num_classes=10):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for events, targets in dataloader:
            events, targets = events.to(device), targets.to(device)
            
            outputs = model(events)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(dataloader), 100. * correct / total