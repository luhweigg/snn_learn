import os
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models.nmnist_snn import NMNIST_SNN
from src.models.cifar10_snn import CIFAR10_SNN
from src.models.dvs_gesture_snn import DVSGesture_SNN
from src.data_loaders.nmnist_loader import get_nmnist_loaders
from src.data_loaders.cifar10_dvs_loader import get_cifar10_loaders
from src.data_loaders.dvs_gesture_loader import get_dvs_gesture_loaders
from src.engine.trainer import train_one_epoch, evaluate
from src.engine.quantization import quantize_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nmnist', choices=['nmnist', 'cifar10', 'dvs_gesture'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Dataset: {args.dataset}")

    if args.dataset == 'nmnist':
        train_loader, test_loader = get_nmnist_loaders(args.batch_size)
        model = NMNIST_SNN().to(device)
    elif args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(args.batch_size)
        model = CIFAR10_SNN().to(device)
    elif args.dataset == 'dvs_gesture':
        train_loader, test_loader = get_dvs_gesture_loaders(args.batch_size)
        model = DVSGesture_SNN().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    base_path = os.path.join(args.save_dir, f"{args.dataset}_base.pth")
    torch.save(model.state_dict(), base_path)
    print(f"\nPoids standards sauvegardés dans : {base_path}")

    print("Application de la quantification post-entrainement (8-bits)...")
    quantize_weights(model, num_bits=8)
    
    quant_loss, quant_acc = evaluate(model, test_loader, criterion, device)
    print(f"Performances apres quantification - Loss: {quant_loss:.4f}, Acc: {quant_acc:.2f}%")

    quantized_path = os.path.join(args.save_dir, f"{args.dataset}_quantized.pth")
    torch.save(model.state_dict(), quantized_path)
    print(f"Poids quantifiés sauvegardés dans : {quantized_path}")

if __name__ == "__main__":
    main()