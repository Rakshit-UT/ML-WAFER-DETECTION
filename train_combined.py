"""
MobileNetV3-Small Training for COMBINED Multi-Domain Defect Classification
12 classes from: Carinthia SEM, WM-811K, MixedWM38, SD-Saliency, DeepPCB
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

# Config
DATA_DIR = Path('/home/ml/CAS/combined_dataset')
OUTPUT_DIR = Path('/home/ml/CAS/training_combined')
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 100
PATIENCE = 15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Focal Loss for imbalanced classes
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / 'val', transform=val_transform)
    test_dataset = datasets.ImageFolder(DATA_DIR / 'test', transform=val_transform)
    
    # Weighted sampling for imbalanced classes
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = class_weights[train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes

class MobileNetV3SmallGrayscale(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        
        # Grayscale input
        original_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            self.mobilenet.features[0][0].weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        # Classifier
        in_features = self.mobilenet.classifier[3].in_features
        self.mobilenet.classifier[3] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.mobilenet(x)

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating', leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, np.array(all_preds), np.array(all_labels)

def plot_results(y_true, y_pred, class_names, history, output_dir):
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix (12 Classes)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlGn(per_class_acc)
    bars = ax.bar(class_names, per_class_acc * 100, color=colors, edgecolor='black')
    ax.axhline(y=90, color='green', linestyle='--', label='90% Target')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Class')
    ax.set_title('Per-Class Accuracy (Combined Dataset)')
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc*100:.1f}%', ha='center', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_accuracy.png', dpi=150)
    plt.close()
    
    # Training history
    epochs = range(1, len(history['train_acc']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax1.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax2.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close()

def train_model():
    print("=" * 60)
    print("MobileNetV3-Small | 12-Class Combined Dataset")
    print("=" * 60)
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    
    model = MobileNetV3SmallGrayscale(num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0
    patience_counter = 0
    
    print("\nTraining...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
            improved = " ★"
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{improved}")
        
        if patience_counter >= PATIENCE:
            print(f"\n⚡ Early stopping at epoch {epoch+1}")
            break
    
    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION")
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth'))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion)
    
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
    print(f"📊 Best Val Accuracy: {best_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    plot_results(test_labels, test_preds, class_names, history, OUTPUT_DIR)
    
    results = {
        'test_accuracy': test_acc,
        'best_val_accuracy': best_acc,
        'classes': list(class_names),
        'num_classes': num_classes,
        'model_params': total_params,
        'epochs_trained': len(history['train_acc']),
        'timestamp': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), OUTPUT_DIR / 'final_model.pth')
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    return test_acc

if __name__ == "__main__":
    acc = train_model()
    print(f"\n{'='*60}")
    print(f"COMPLETE! Test Accuracy: {acc:.2f}%")
    print("=" * 60)
