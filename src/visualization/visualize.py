import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_training_metrics(train_losses, accuracies_per_exit, exit_distributions):
    """
    Visualize training progress with multiple metrics.
    Shows loss curve, accuracy per exit point, and exit point distribution.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # Plot accuracy for each exit
    for i, accs in enumerate(accuracies_per_exit):
        ax2.plot(accs, label=f'Exit {i+1}')
    ax2.set_title('Accuracy per Exit Point')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Plot exit distribution
    exits = ['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4']
    ax3.bar(exits, exit_distributions)
    ax3.set_title('Exit Point Distribution')
    ax3.set_ylabel('Percentage of Samples')
    ax3.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

def visualize_class_exits(model, agent, test_loader, num_samples=50):
    """
    Visualize which classes tend to exit at which points.
    Creates a heatmap showing the relationship between classes and exit points.
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')

    exit_stats = np.zeros((10, 4))  # 10 classes, 4 exit points
    class_counts = np.zeros(10)

    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            for i in range(len(data)):
                sample = data[i:i+1]
                label = target[i].item()

                # Get predictions from all exits
                exits = model(sample, return_all_exits=True)

                # Determine exit point using agent
                state = F.softmax(exits[0], dim=1).max(1)[0].cpu().numpy()
                exit_point = 0

                while True:
                    action = agent.get_action(state)
                    if action == 0 or exit_point == len(exits) - 1:
                        break
                    exit_point += 1
                    state = F.softmax(exits[exit_point], dim=1).max(1)[0].cpu().numpy()

                exit_stats[label, exit_point] += 1
                class_counts[label] += 1

                if np.sum(class_counts) >= num_samples * 10:
                    break
            if np.sum(class_counts) >= num_samples * 10:
                break

    # Normalize by class counts
    exit_stats = exit_stats / class_counts[:, np.newaxis] * 100

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(exit_stats,
                xticklabels=['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4'],
                yticklabels=classes,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd')
    plt.title('Exit Point Distribution by Class (%)')
    plt.xlabel('Exit Point')
    plt.ylabel('Class')
    plt.show()

def visualize_exit_decisions(model, agent, test_loader, num_samples=50):
    """
    Visualize sample images and their exit points.
    Shows confidence levels and actual vs predicted classes.
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')

    model.eval()
    images, labels = next(iter(test_loader))
    num_samples = min(num_samples, len(images))  # Ensure we don't exceed available images
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    # Calculate grid dimensions - 5 columns, 10 rows for 50 images
    nrows = 10
    ncols = 5
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 40))  # Increased figure size for visibility
    axes = axes.ravel()  # Flatten the array of axes

    with torch.no_grad():
        for i, (image, label) in enumerate(zip(images, labels)):
            # Get predictions from all exits
            exits = model(image.unsqueeze(0), return_all_exits=True)

            # Get confidence scores and determine exit point
            confidences = [F.softmax(exit, dim=1).max(1)[0].item() for exit in exits]

            state = confidences[0]
            exit_point = 0
            while True:
                action = agent.get_action([state])
                if action == 0 or exit_point == len(exits) - 1:
                    break
                exit_point += 1
                state = confidences[exit_point]

            # Get prediction
            pred = exits[exit_point].argmax(1).item()

            # Plot image
            img = image.cpu().numpy().transpose(1, 2, 0)
            # Denormalize image
            img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
            img = np.clip(img, 0, 1)

            axes[i].imshow(img)
            axes[i].axis('off')
            title = f'True: {classes[label]}\nPred: {classes[pred]}\nExit: {exit_point+1}\nConf: {confidences[exit_point]:.2f}'
            # Color code the title based on correct/incorrect prediction
            color = 'green' if pred == label else 'red'
            axes[i].set_title(title, color=color)

    plt.tight_layout()
    plt.show()
    
def plot_confidence_distributions(model, test_loader):
    """
    Plot confidence distribution at each exit point.
    Shows how confidence levels vary across different exits.
    """
    confidences_per_exit = [[] for _ in range(4)]

    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            exits = model(data, return_all_exits=True)

            for i, exit in enumerate(exits):
                confs = F.softmax(exit, dim=1).max(1)[0].cpu().numpy()
                confidences_per_exit[i].extend(confs)

    plt.figure(figsize=(12, 6))
    for i, confs in enumerate(confidences_per_exit):
        plt.hist(confs, bins=50, alpha=0.5, label=f'Exit {i+1}')

    plt.title('Confidence Distribution at Each Exit')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_exit_analysis(metrics):
    """
    Creates visualizations of the exit analysis results.
    """
    accuracies = metrics['accuracies']
    classes = metrics['classes']

    # Create accuracy comparison plot
    plt.figure(figsize=(15, 8))
    x = np.arange(len(classes))
    width = 0.2

    for i in range(4):
        plt.bar(x + i*width, accuracies[i] * 100, width,
                label=f'Exit {i+1}')

    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy at Each Exit')
    plt.xticks(x + width*1.5, classes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create heatmap of accuracies
    plt.figure(figsize=(12, 8))
    sns.heatmap(accuracies * 100,
                xticklabels=classes,
                yticklabels=[f'Exit {i+1}' for i in range(4)],
                annot=True,
                fmt='.1f',
                cmap='YlOrRd')
    plt.title('Accuracy Heatmap: Exits vs Classes')
    plt.tight_layout()
    plt.show()