# main_cosine_annealing.py

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
                model.train()  
                dataloader = train_loader
            else:
                model.eval()   
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

    # Experiment 1: Constant Learning Rate
    print('Experiment 1: Training with constant learning rate')
    # Initialize the model
    model_const_lr = MobileNet(num_classes=100, sigmoid_block_ind=[])  
    model_const_lr = model_const_lr.to(device)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_const = optim.SGD(model_const_lr.parameters(), lr=initial_lr, momentum=0.9)  # Added momentum=0.9

    # Train the model without scheduler
    trained_model_const_lr, train_loss_const, val_loss_const, train_acc_const, val_acc_const, lr_history_const = train_model(
        model_const_lr,
        dataloaders,
        criterion,
        optimizer_const,
        scheduler=None,
        num_epochs=num_epochs,
        exp_type='constant_lr'
    )

    # Save the trained model
    model_path_const = f'./models/mobilenet_const_lr_{initial_lr}.pth'
    torch.save(trained_model_const_lr.state_dict(), model_path_const)
    print(f'Model saved to {model_path_const}')

    # Plot and save the training curves
    fig_name_const = f'training_curve_const_lr_{initial_lr}.png'
    plot_loss_acc(train_loss_const, val_loss_const, train_acc_const, val_acc_const, fig_name_const)
    print(f'Training curves saved to ./diagram/{fig_name_const}')

    # Plot learning rate
    plot_lr(lr_history_const, f'learning_rate_const_{initial_lr}.png')

    # Report the final losses and accuracies
    final_train_loss_const = train_loss_const[-1]
    final_val_loss_const = val_loss_const[-1]
    final_train_acc_const = train_acc_const[-1]
    final_val_acc_const = val_acc_const[-1]

    print(f'Final Training Loss (Constant LR): {final_train_loss_const:.4f}')
    print(f'Final Validation Loss (Constant LR): {final_val_loss_const:.4f}')
    print(f'Final Training Accuracy (Constant LR): {final_train_acc_const:.4f}')
    print(f'Final Validation Accuracy (Constant LR): {final_val_acc_const:.4f}')
    print('-' * 50)

    # Experiment 2: Cosine Annealing Learning Rate
    print('Experiment 2: Training with cosine annealing learning rate schedule')
    # Initialize the model
    model_cosine_lr = MobileNet(num_classes=100, sigmoid_block_ind=[])
    model_cosine_lr = model_cosine_lr.to(device)

    # Define optimizer and scheduler
    optimizer_cosine = optim.SGD(model_cosine_lr.parameters(), lr=initial_lr, momentum=0.9)  
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_cosine, T_max=num_epochs, eta_min=0)

    # Train the model with cosine annealing scheduler
    trained_model_cosine_lr, train_loss_cosine, val_loss_cosine, train_acc_cosine, val_acc_cosine, lr_history_cosine = train_model(
        model_cosine_lr,
        dataloaders,
        criterion,
        optimizer_cosine,
        scheduler_cosine,
        num_epochs=num_epochs,
        exp_type='cosine_annealing'
    )

    # Save the trained model
    model_path_cosine = f'./models/mobilenet_cosine_lr_{initial_lr}.pth'
    torch.save(trained_model_cosine_lr.state_dict(), model_path_cosine)
    print(f'Model saved to {model_path_cosine}')

    # Plot and save the training curves
    fig_name_cosine = f'training_curve_cosine_lr_{initial_lr}.png'
    plot_loss_acc(train_loss_cosine, val_loss_cosine, train_acc_cosine, val_acc_cosine, fig_name_cosine)
    print(f'Training curves saved to ./diagram/{fig_name_cosine}')

    # Plot learning rate
    plot_lr(lr_history_cosine, f'learning_rate_cosine_{initial_lr}.png')

    # Report the final losses and accuracies
    final_train_loss_cosine = train_loss_cosine[-1]
    final_val_loss_cosine = val_loss_cosine[-1]
    final_train_acc_cosine = train_acc_cosine[-1]
    final_val_acc_cosine = val_acc_cosine[-1]

    print(f'Final Training Loss (Cosine Annealing LR): {final_train_loss_cosine:.4f}')
    print(f'Final Validation Loss (Cosine Annealing LR): {final_val_loss_cosine:.4f}')
    print(f'Final Training Accuracy (Cosine Annealing LR): {final_train_acc_cosine:.4f}')
    print(f'Final Validation Accuracy (Cosine Annealing LR): {final_val_acc_cosine:.4f}')
    print('-' * 50)

if __name__ == '__main__':
    main()
