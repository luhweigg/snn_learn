import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models import NMNIST_SNN, CIFAR10_SNN, DVSGesture_SNN
from src.data_loaders import get_nmnist_loaders, get_cifar10_loaders, get_dvs_gesture_loaders

def build_components(dataset, batch_size, time_steps, lr, epochs, device):
    """
    Build model, data loaders, optimizer, scheduler, criterion, and scaler based on the specified dataset.
    """
    if dataset == 'nmnist':
        train_loader, test_loader = get_nmnist_loaders(batch_size, time_steps)
        model = NMNIST_SNN().to(device)
    elif dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(batch_size, time_steps)
        model = CIFAR10_SNN().to(device)
    elif dataset == 'dvs_gesture':
        train_loader, test_loader = get_dvs_gesture_loaders(batch_size, time_steps)
        model = DVSGesture_SNN().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    return model, train_loader, test_loader, optimizer, scheduler, criterion, scaler