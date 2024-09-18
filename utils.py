import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, fig_name):
    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim([0, max_loss + 1])
    lns1 = ax1.plot(x, train_loss, color='y', linestyle='-', marker='o', label='Train Loss', linewidth=1.5)
    lns2 = ax1.plot(x, val_loss, color='g', linestyle='-', marker='s', label='Validation Loss', linewidth=1.5)
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1.1])
    lns3 = ax2.plot(x, train_acc, color='b', linestyle='-', marker='^', label='Train Accuracy', linewidth=1.5)
    lns4 = ax2.plot(x, val_acc, color='r', linestyle='-', marker='*', label='Validation Accuracy', linewidth=1.5)
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combine legends from both axes
    lns = lns1 + lns2 + lns3 + lns4
    labels = [l.get_label() for l in lns]
    ax2.legend(lns, labels, loc='best')

    # fig.tight_layout()
    plt.title('Training and Validation Loss and Accuracy')
    plt.grid(True)

    os.makedirs('./diagram', exist_ok=True)
    plt.savefig(os.path.join('./diagram', fig_name))
    plt.close(fig)  # Close the figure to free memory

    # Save data for future use if needed
    np.savez(
        os.path.join('./diagram', fig_name.replace('.png', '.npz')),
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc
    )

def plot_lr(lr, fig_name):
    plt.figure()
    plt.plot(range(len(lr)), lr, color='r', linestyle='-', marker='o', linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curve')
    plt.grid(True)

    os.makedirs('./diagram', exist_ok=True)
    plt.savefig(os.path.join('./diagram', fig_name))
    plt.close()  # Close the figure to free memory

    np.savez(
        os.path.join('./diagram', fig_name.replace('.png', '.npz')),
        lr=lr
    )
