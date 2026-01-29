import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntelImageDataset(Dataset):
    """Custom Dataset for Intel Image Classification"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


class CustomCNN(nn.Module):
    """Custom CNN architecture for image classification"""
    
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, early_stopping_patience=10):
    """Train the model with early stopping"""
    
    best_val_acc = 0.0
    best_model_weights = None
    patience_counter = 0
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    lr_history = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_labels, val_preds)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_epoch_loss)
        for param_group in optimizer.param_groups:
            lr_history.append(param_group['lr'])
        
        # Early stopping and model checkpointing
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
            print(f'✓ New best validation accuracy: {best_val_acc:.4f}')
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{early_stopping_patience}')
        
        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model, train_losses, train_accs, val_losses, val_accs, lr_history


def evaluate_model(model, test_loader, class_names):
    """Evaluate the model on test set"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    print(f'\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
    
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return test_acc


def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/muhammad_adib/imgclass_cnn/training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved as 'training_history.png'")


def tta_inference(model, images, device):
    model.eval()
    batch_size, c, h, w = images.size()
    
    # 1. Define our augmentations
    # We use horizontal flip and slight shifts/crops
    with torch.no_grad():
        # Original
        out1 = model(images)
        
        # Horizontal Flip
        out2 = model(torch.flip(images, dims=[3]))
        
        # Slight Shift Left/Up (using roll for speed)
        out3 = model(torch.roll(images, shifts=(5, 5), dims=(2, 3)))
        
        # Slight Shift Right/Down
        out4 = model(torch.roll(images, shifts=(-5, -5), dims=(2, 3)))

        # 2. Convert all to probabilities (Softmax) before averaging
        # This is the secret to why TTA usually outperforms single-pass
        p1 = F.softmax(out1, dim=1)
        p2 = F.softmax(out2, dim=1)
        p3 = F.softmax(out3, dim=1)
        p4 = F.softmax(out4, dim=1)
        
        # 3. Average the probabilities
        avg_probs = (p1 + p2 + p3 + p4) / 4
        
        return avg_probs
    
from sklearn.metrics import accuracy_score

def evaluate_with_tta(model, test_loader, device):
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Get TTA probabilities
            probs = tta_inference(model, images, device)
            
            # Get the highest probability class
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Final Scikit-Learn Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Final TTA Accuracy: {acc * 100:.2f}%")
    return acc




def main():
    # Image size - must be consistent across train and validation
    IMG_SIZE = 150  # Use same size for both

    # 1. Define your two different sets of transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Same size as training!
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load the base dataset (no transform yet)
    path = '/home/muhammad_adib/imgclass_cnn/Data'
    base_dataset = datasets.ImageFolder(root=path)

    # 3. Get stratified indices
    # 70-15-15 split
    train_val_idx, test_idx = train_test_split(
        np.arange(len(base_dataset.targets)),
        test_size=0.15, # 15% for test
        stratify=base_dataset.targets,
        random_state=42
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1765, # ~15% of original data for validation
        stratify=[base_dataset.targets[i] for i in train_val_idx],
        random_state=42
    )

    train_dataset = ApplyTransform(Subset(base_dataset, train_idx), transform=train_transform)
    val_dataset = ApplyTransform(Subset(base_dataset, val_idx), transform=val_transform)
    test_dataset = ApplyTransform(Subset(base_dataset, test_idx), transform=val_transform)

    # 5. DataLoaders - Using num_workers=0 to avoid BrokenPipeError
    # For RTX 4070, GPU is fast enough that this won't be a bottleneck
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Get class names
    class_names = sorted(os.listdir(path))
    num_classes = len(class_names)
    

    # Initialize model
    print(f"\nInitializing Custom CNN model...")
    model = CustomCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Train the model
    model, train_losses, train_accs, val_losses, val_accs, lr_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=50, early_stopping_patience=6
    )
    
    # Prepare data as a dictionary
    history_dict = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'lr': lr_history
    }

    # Create DataFrame
    history_df = pd.DataFrame(history_dict)

    # Save to CSV
    history_df.to_csv('/home/muhammad_adib/imgclass_cnn/training_history.csv', index_label='epoch')
    print("Training history saved to 'training_history.csv'")
    # Evaluate the model

    test_acc = evaluate_model(model, test_loader, class_names)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    print("\nEvaluating with TTA on test set...")
    tta_acc = evaluate_with_tta(model, test_loader, device)
    print(f"TTA Test Accuracy: {tta_acc:.4f} ({tta_acc*100:.2f}%)")
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'test_accuracy': test_acc,
        'tta_accuracy': tta_acc
    }, '/home/muhammad_adib/imgclass_cnn/intel_cnn_model.pth')
    print("\nModel saved as 'intel_cnn_model.pth'")

if __name__ == '__main__':
    main()