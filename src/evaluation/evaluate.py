import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def evaluate_model(model, agent, test_loader):
    """
    Enhanced evaluation with confidence calibration, per-class metrics,
    and detailed analysis of exit decisions.
    """
    print("Evaluating model...")
    model.eval()

    # Tracking metrics
    correct = 0
    total = 0
    exit_counts = [0] * 4
    class_correct = [0] * 10
    class_total = [0] * 10
    compute_saved = 0
    confidence_stats = [[] for _ in range(4)]
    exit_decision_changes = 0  # Track when predictions change between exits

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')

    def get_calibrated_confidence(logits, temperature=1.0):
        """Apply temperature scaling for better calibrated confidence scores"""
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)
        return probs.max(1)[0]

    def analyze_prediction_stability(exits, confidences):
        """Analyze how predictions change across exits"""
        predictions = [exit.argmax(dim=1) for exit in exits]
        changes = sum(1 for i in range(1, len(predictions))
                     if predictions[i] != predictions[i-1])
        return changes, predictions

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            batch_size = len(data)

            for i in range(batch_size):
                sample = data[i:i+1]
                sample_target = target[i:i+1]

                # Get predictions from all exits
                exits = model(sample, return_all_exits=True)

                # Get calibrated confidences
                confidences = [get_calibrated_confidence(exit) for exit in exits]

                # Analyze prediction stability
                pred_changes, predictions = analyze_prediction_stability(exits, confidences)

                # Initialize state with first exit's confidence
                state = confidences[0].cpu().numpy()
                exit_point = 0
                prev_pred = None

                # Let agent make exit decisions
                while True:
                    action = agent.get_action(state)

                    # Enhanced exit decision logic
                    current_pred = exits[exit_point].argmax(dim=1)
                    current_conf = confidences[exit_point].item()

                    # Store confidence statistics
                    confidence_stats[exit_point].append(current_conf)

                    # Track prediction changes
                    if prev_pred is not None and current_pred != prev_pred:
                        exit_decision_changes += 1

                    # Exit conditions with adaptive thresholds
                    if action == 0:
                        # Check if confidence is high enough for this exit
                        min_conf = 0.98 - (exit_point * 0.05)  # Adaptive threshold
                        if current_conf >= min_conf:
                            break
                        elif exit_point < len(exits) - 1:
                            exit_point += 1
                            state = confidences[exit_point].cpu().numpy()
                            prev_pred = current_pred
                            continue

                    if exit_point == len(exits) - 1:
                        break

                    exit_point += 1
                    state = confidences[exit_point].cpu().numpy()
                    prev_pred = current_pred

                # Record exit point
                exit_counts[exit_point] += 1

                # Check accuracy
                final_prediction = exits[exit_point].argmax(dim=1)
                correct_prediction = final_prediction.eq(sample_target).item()
                correct += correct_prediction
                total += 1

                # Update per-class accuracy
                class_idx = sample_target.item()
                class_correct[class_idx] += correct_prediction
                class_total[class_idx] += 1

                # Calculate compute saved
                compute_saved += (3 - exit_point) / 3

    # Calculate overall metrics
    accuracy = 100. * correct / total
    exit_distribution = [count/total*100 for count in exit_counts]
    compute_saved = compute_saved / total * 100

    # Calculate confidence statistics
    confidence_means = [np.mean(confs) if confs else 0 for confs in confidence_stats]
    confidence_stds = [np.std(confs) if confs else 0 for confs in confidence_stats]

    # Print comprehensive evaluation results
    print(f"\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Exit Distribution: Exit1={exit_distribution[0]:.1f}%, "
          f"Exit2={exit_distribution[1]:.1f}%, "
          f"Exit3={exit_distribution[2]:.1f}%, "
          f"Exit4={exit_distribution[3]:.1f}%")
    print(f"Compute Saved: {compute_saved:.1f}%")
    print(f"Prediction Changes Between Exits: {exit_decision_changes}")

    print("\nConfidence Statistics:")
    for i in range(4):
        print(f"Exit {i+1}: Mean={confidence_means[i]:.3f}, Std={confidence_stds[i]:.3f}")

    print("\nPer-class Accuracy:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {class_acc:.2f}%')

    # Calculate effectiveness score
    effectiveness = accuracy * (compute_saved/100)  # Balance between accuracy and efficiency
    print(f"\nEffectiveness Score: {effectiveness:.2f}")

    return {
        'accuracy': accuracy,
        'exit_distribution': exit_distribution,
        'compute_saved': compute_saved,
        'confidence_stats': {
            'means': confidence_means,
            'stds': confidence_stds
        },
        'class_accuracies': [100 * correct / total for correct, total in zip(class_correct, class_total)],
        'effectiveness': effectiveness,
        'exit_decision_changes': exit_decision_changes
    }

def analyze_per_class_exits(model, test_loader, device):
    """
    Analyzes the performance of each exit point for each class.
    Returns detailed metrics about accuracy and confidence per class per exit.
    """
    model.eval()

    # Initialize tracking matrices
    num_classes = 10
    num_exits = 4

    # Track correct predictions and total instances for each class at each exit
    correct_per_class_exit = torch.zeros((num_exits, num_classes))
    total_per_class_exit = torch.zeros((num_exits, num_classes))

    # Track confidence scores
    confidence_sums = torch.zeros((num_exits, num_classes))

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc='Analyzing exits'):
            data, targets = data.to(device), targets.to(device)

            # Get predictions from all exits
            exits = model(data, return_all_exits=True)

            # Analyze each exit point
            for exit_idx, exit_output in enumerate(exits):
                # Get predictions and confidence scores
                confidences, predictions = torch.max(F.softmax(exit_output, dim=1), dim=1)

                # Update metrics for each class
                for class_idx in range(num_classes):
                    # Create mask for current class
                    class_mask = targets == class_idx

                    # Count total instances of this class
                    total_per_class_exit[exit_idx][class_idx] += class_mask.sum().item()

                    # Count correct predictions for this class
                    correct_predictions = (predictions == targets) & class_mask
                    correct_per_class_exit[exit_idx][class_idx] += correct_predictions.sum().item()

                    # Sum confidence scores for this class
                    confidence_sums[exit_idx][class_idx] += confidences[class_mask].sum().item()

    # Calculate accuracies and average confidences
    accuracies = correct_per_class_exit / total_per_class_exit
    avg_confidences = confidence_sums / total_per_class_exit

    # Print detailed analysis
    print("\nPer-Class Accuracy at Each Exit Point:")
    print("-" * 50)

    for class_idx, class_name in enumerate(classes):
        print(f"\n{class_name.upper()}:")
        for exit_idx in range(num_exits):
            acc = accuracies[exit_idx][class_idx] * 100
            conf = avg_confidences[exit_idx][class_idx]
            total = total_per_class_exit[exit_idx][class_idx]
            print(f"Exit {exit_idx + 1}: "
                  f"Accuracy = {acc:.2f}%, "
                  f"Avg Confidence = {conf:.3f}, "
                  f"Samples = {total:.0f}")

    # Return detailed metrics for visualization
    return {
        'accuracies': accuracies.numpy(),
        'confidences': avg_confidences.numpy(),
        'totals': total_per_class_exit.numpy(),
        'classes': classes
    }