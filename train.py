import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CIFAR10_CNN
from dataset import CIFAR10Dataset
from torchsummary import summary    


def train_model(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{accuracy:.2f}%'
        })
    
    accuracy = 100. * correct / total
    return running_loss / len(train_loader), accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{accuracy:.2f}%'
            })
    
    accuracy = 100. * correct / total
    return running_loss / len(test_loader), accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = CIFAR10Dataset(train=True)
    test_dataset = CIFAR10Dataset(train=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=True, num_workers=4
    )
    
    # Initialize model, criterion, and optimizer
    model = CIFAR10_CNN().to(device)
    summary(model, (3, 32, 32))

    criterion = nn.CrossEntropyLoss()
    # Using AdamW instead of SGD for better convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=50,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
    )
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
    
    # Training loop
    best_acc = 0.0
    num_epochs = 50
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Test
        test_loss, test_acc = test_model(
            model, test_loader, criterion, device
        )
        
        # Print epoch results
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'train_acc': train_acc,
            #     'test_acc': test_acc,
            # }, 'best_model.pth')
            print(f'New best model saved with test accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()