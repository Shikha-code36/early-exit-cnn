import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def pretrain_cnn(model, train_loader, val_loader=None, num_epochs=20):
    """
    Enhanced pre-training with dynamic exit weights, validation monitoring,
    and improved optimization strategy.
    """
    print("Pre-training CNN...")
    model.train()

    # Training metrics
    train_losses = []
    accuracies_per_exit = [[] for _ in range(4)]
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    # Optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6  # Adjusted for 50 epochs
    )

    def get_dynamic_weights(epoch, num_epochs):
        """Calculate dynamic weights for exits based on training progress"""
        progress = epoch / num_epochs
        # Gradually shift importance to later exits
        weights = torch.tensor([
            0.4 * (1 - progress),  # Decrease weight of first exit
            0.2 + 0.1 * progress,  # Slightly increase second exit
            0.2 + 0.2 * progress,  # Moderately increase third exit
            0.2 + 0.3 * progress   # Significantly increase final exit
        ]).to(device)
        return weights / weights.sum()  # Normalize weights

    def validate_model(model, val_loader):
        """Evaluate model performance on validation set"""
        model.eval()
        val_losses = []
        correct_per_exit = [0] * 4
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                exits = model(data, return_all_exits=True)

                for i, exit_output in enumerate(exits):
                    loss = F.cross_entropy(exit_output, target)
                    val_losses.append(loss.item())
                    pred = exit_output.argmax(dim=1)
                    correct_per_exit[i] += pred.eq(target).sum().item()
                total += len(target)

        # Calculate accuracies
        accuracies = [100 * correct / total for correct in correct_per_exit]
        avg_loss = sum(val_losses) / len(val_losses)

        model.train()
        return avg_loss, accuracies

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = [0] * 4
        total_samples = 0

        # Get dynamic weights for this epoch
        exit_weights = get_dynamic_weights(epoch, num_epochs)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Get predictions from all exits
            exits = model(data, return_all_exits=True)

            # Compute weighted loss with dynamic weights
            loss = 0
            for i, (exit_output, weight) in enumerate(zip(exits, exit_weights)):
                exit_loss = F.cross_entropy(exit_output, target)
                loss += weight * exit_loss

                # Track accuracy
                pred = exit_output.argmax(dim=1)
                correct_predictions[i] += pred.eq(target).sum().item()

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_samples += len(data)

            # Update progress bar with metrics
            avg_loss = epoch_loss/(batch_idx+1)
            accuracies = [correct/total_samples * 100 for correct in correct_predictions]
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc1': f'{accuracies[0]:.1f}%',
                'acc2': f'{accuracies[1]:.1f}%',
                'acc3': f'{accuracies[2]:.1f}%',
                'acc4': f'{accuracies[3]:.1f}%'
            })

        # Validation phase
        if val_loader is not None:
            val_loss, val_accuracies = validate_model(model, val_loader)
            print(f"\nValidation - Loss: {val_loss:.4f}, "
                  f"Accuracies: {[f'{acc:.1f}%' for acc in val_accuracies]}")

            # Early stopping check
            current_val_acc = val_accuracies[-1]  # Use final exit accuracy
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

        # Update learning rate
        scheduler.step()

        # Store metrics
        train_losses.append(avg_loss)
        for i in range(4):
            accuracies_per_exit[i].append(accuracies[i])

    print("CNN pre-training completed!")
    return train_losses, accuracies_per_exit