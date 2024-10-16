# weight_decay.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import copy
from data import get_train_valid_loader, get_test_loader
from mobilenet import MobileNet
from utils import plot_loss_acc, plot_lr

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, exp_type):
    train_loader = dataloaders['train']
    valid_loader = dataloaders['val']

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    lr_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}/{num_epochs} - LR: {current_lr}')
        print('-' * 30)

        lr_history.append(current_lr)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it has better accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            # Record loss and accuracy
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

        # Step the scheduler at the end of the epoch
        if scheduler is not None:
            scheduler.step()

        print()

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, lr_history

def main():
    # Hyperparameters and settings
    data_dir = './data'
    batch_size = 128
    num_epochs = 300
    initial_lr = 0.05  
    weight_decays = [5e-4, 1e-4]
    random_seed = 0

    # Ensure deterministic behavior
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Create directory to save models and results
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./diagram', exist_ok=True)

    # Data augmentation: random cropping and random horizontal flip
    augment = True

    # Load data
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = get_train_valid_loader(
        data_dir,
        batch_size,
        augment,
        random_seed,
        valid_size=0.2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Experiment with different weight decay coefficients
    for wd in weight_decays:
        print(f'Experiment: Training with weight decay λ = {wd}')
        # Initialize the model
        model = MobileNet(num_classes=100,sigmoid_block_ind=[])  # CIFAR-100 has 100 classes
        model = model.to(device)

        # Define loss function (criterion) and optimizer with weight decay
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, weight_decay=wd)

        # Define the cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

        # Train the model
        trained_model, train_loss, val_loss, train_acc, val_acc, lr_history = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            exp_type=f'weight_decay_{wd}'
        )

        # Save the trained model
        model_path = f'./models/mobilenet_wd_{wd}_lr_{initial_lr}.pth'
        torch.save(trained_model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

        # Plot and save the training curves
        fig_name = f'training_curve_wd_{wd}_lr_{initial_lr}.png'
        plot_loss_acc(train_loss, val_loss, train_acc, val_acc, fig_name)
        print(f'Training curves saved to ./diagram/{fig_name}')

        # Plot learning rate
        plot_lr(lr_history, f'learning_rate_wd_{wd}_lr_{initial_lr}.png')

        # Report the final losses and accuracies
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        final_train_acc = train_acc[-1]
        final_val_acc = val_acc[-1]

        print(f'Final Training Loss (Weight Decay λ={wd}): {final_train_loss:.4f}')
        print(f'Final Validation Loss (Weight Decay λ={wd}): {final_val_loss:.4f}')
        print(f'Final Training Accuracy (Weight Decay λ={wd}): {final_train_acc:.4f}')
        print(f'Final Validation Accuracy (Weight Decay λ={wd}): {final_val_acc:.4f}')
        print('-' * 50)

        # Evaluate on the hold-out test set
        test_loader = get_test_loader(
            data_dir,
            batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model.eval()  # Set model to evaluate mode
        test_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)

        test_acc = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy (Weight Decay λ={wd}): {test_acc:.4f}')
        print('=' * 50)

if __name__ == '__main__':
    main()
