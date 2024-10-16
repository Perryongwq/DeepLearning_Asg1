# activationfunction.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from data import get_train_valid_loader, get_test_loader
from mobilenet import MobileNet
from utils import plot_loss_acc, plot_lr
import matplotlib.pyplot as plt
import numpy as np

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
    grad_norm_history = []  # To store gradient norms

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

                        # Compute the gradient norm of model.layers[8].conv1.weight
                        grad_norm = model.layers[8].conv1.weight.grad.norm(2).item()
                        grad_norm_history.append(grad_norm)

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
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, lr_history, grad_norm_history

def plot_grad_norm(grad_norm_history, fig_name):
    plt.figure()
    plt.plot(grad_norm_history, color='m', linestyle='-', marker='o', markersize=4, linewidth=1.5)
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm of model.layers[8].conv1.weight over Training Steps')
    plt.grid(True)
    os.makedirs('./diagram', exist_ok=True)
    plt.savefig(os.path.join('./diagram', fig_name))
    plt.close()

    # Save gradient norms for future use if needed
    np.savez(
        os.path.join('./diagram', fig_name.replace('.png', '.npz')),
        grad_norm_history=grad_norm_history
    )

def main():
    # Hyperparameters and settings
    data_dir = './data'
    batch_size = 128
    num_epochs = 300
    initial_lr = 0.05  # Replace with the best learning rate identified earlier
    weight_decay = 5e-4  # Replace with the best weight decay identified earlier
    random_seed = 42  # For reproducibility

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

    # Initialize the model with sigmoid activation in blocks 4-10
    sigmoid_block_ind = [4,5,6,7,8,9,10]
    model_sigmoid = MobileNet(num_classes=100, sigmoid_block_ind=sigmoid_block_ind)
    model_sigmoid = model_sigmoid.to(device)

    # Define loss function (criterion) and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer_sigmoid = optim.SGD(model_sigmoid.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)

    # Define the cosine annealing scheduler
    scheduler_sigmoid = optim.lr_scheduler.CosineAnnealingLR(optimizer_sigmoid, T_max=num_epochs, eta_min=0)

    # Train the model with sigmoid activation
    trained_model_sigmoid, train_loss_sigmoid, val_loss_sigmoid, train_acc_sigmoid, val_acc_sigmoid, lr_history_sigmoid, grad_norm_history = train_model(
        model_sigmoid,
        dataloaders,
        criterion,
        optimizer_sigmoid,
        scheduler_sigmoid,
        num_epochs,
        exp_type='sigmoid_activation'
    )

    # Save the trained model
    model_path_sigmoid = f'./models/mobilenet_sigmoid_activation.pth'
    torch.save(trained_model_sigmoid.state_dict(), model_path_sigmoid)
    print(f'Model with sigmoid activation saved to {model_path_sigmoid}')

    # Plot and save the training curves
    fig_name_sigmoid = f'training_curve_sigmoid_activation.png'
    plot_loss_acc(train_loss_sigmoid, val_loss_sigmoid, train_acc_sigmoid, val_acc_sigmoid, fig_name_sigmoid)
    print(f'Training curves saved to ./diagram/{fig_name_sigmoid}')

    # Plot learning rate
    plot_lr(lr_history_sigmoid, f'learning_rate_sigmoid_activation.png')

    # Plot gradient norms
    plot_grad_norm(grad_norm_history, 'grad_norm_layer8_conv1_weight.png')
    print('Gradient norm plot saved to ./diagram/grad_norm_layer8_conv1_weight.png')

    # Report the final losses and accuracies
    final_train_loss_sigmoid = train_loss_sigmoid[-1]
    final_val_loss_sigmoid = val_loss_sigmoid[-1]
    final_train_acc_sigmoid = train_acc_sigmoid[-1]
    final_val_acc_sigmoid = val_acc_sigmoid[-1]

    print(f'Final Training Loss (Sigmoid Activation): {final_train_loss_sigmoid:.4f}')
    print(f'Final Validation Loss (Sigmoid Activation): {final_val_loss_sigmoid:.4f}')
    print(f'Final Training Accuracy (Sigmoid Activation): {final_train_acc_sigmoid:.4f}')
    print(f'Final Validation Accuracy (Sigmoid Activation): {final_val_acc_sigmoid:.4f}')
    print('-' * 50)

    # Compare with the ReLU version
    # Initialize the model with ReLU activation (default)
    model_relu = MobileNet(num_classes=100, sigmoid_block_ind=[])
    model_relu = model_relu.to(device)

    # Define optimizer and scheduler for ReLU model
    optimizer_relu = optim.SGD(model_relu.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler_relu = optim.lr_scheduler.CosineAnnealingLR(optimizer_relu, T_max=num_epochs, eta_min=0)

    # Train the model with ReLU activation
    trained_model_relu, train_loss_relu, val_loss_relu, train_acc_relu, val_acc_relu, lr_history_relu, _ = train_model(
        model_relu,
        dataloaders,
        criterion,
        optimizer_relu,
        scheduler_relu,
        num_epochs,
        exp_type='relu_activation'
    )

    # Save the trained ReLU model
    model_path_relu = f'./models/mobilenet_relu_activation.pth'
    torch.save(trained_model_relu.state_dict(), model_path_relu)
    print(f'Model with ReLU activation saved to {model_path_relu}')

    # Plot and save the training curves for ReLU model
    fig_name_relu = f'training_curve_relu_activation.png'
    plot_loss_acc(train_loss_relu, val_loss_relu, train_acc_relu, val_acc_relu, fig_name_relu)
    print(f'Training curves saved to ./diagram/{fig_name_relu}')

    # Plot learning rate for ReLU model
    plot_lr(lr_history_relu, f'learning_rate_relu_activation.png')

    # Report the final losses and accuracies for ReLU model
    final_train_loss_relu = train_loss_relu[-1]
    final_val_loss_relu = val_loss_relu[-1]
    final_train_acc_relu = train_acc_relu[-1]
    final_val_acc_relu = val_acc_relu[-1]

    print(f'Final Training Loss (ReLU Activation): {final_train_loss_relu:.4f}')
    print(f'Final Validation Loss (ReLU Activation): {final_val_loss_relu:.4f}')
    print(f'Final Training Accuracy (ReLU Activation): {final_train_acc_relu:.4f}')
    print(f'Final Validation Accuracy (ReLU Activation): {final_val_acc_relu:.4f}')
    print('-' * 50)

    # Plot the comparison of training and validation accuracies
    plt.figure()
    plt.plot(train_acc_sigmoid, label='Train Acc (Sigmoid)', linestyle='-', marker='o')
    plt.plot(val_acc_sigmoid, label='Val Acc (Sigmoid)', linestyle='-', marker='s')
    plt.plot(train_acc_relu, label='Train Acc (ReLU)', linestyle='-', marker='^')
    plt.plot(val_acc_relu, label='Val Acc (ReLU)', linestyle='-', marker='*')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('./diagram/accuracy_comparison.png')
    plt.close()
    print('Accuracy comparison plot saved to ./diagram/accuracy_comparison.png')

if __name__ == '__main__':
    main()
