# Early Exit CNN with Reinforcement Learning

This project implements an Early Exit Convolutional Neural Network (EE-CNN) for efficient image classification on the CIFAR-10 dataset. The model uses reinforcement learning to dynamically decide when to exit the network early, optimizing the trade-off between computational efficiency and accuracy.

## Architecture Overview

The system consists of two main components:

1. **Early Exit CNN**: A deep neural network with multiple exit points, allowing predictions at different depths:
   - 4 exit points with increasing complexity
   - Enhanced feature extraction at each stage
   - Batch normalization and dropout for regularization
   - Progressive increase in channel dimensions (64→128→256→512)

2. **DQN Agent**: A reinforcement learning agent that learns when to exit:
   - Makes exit decisions based on confidence scores
   - Balances accuracy and computational efficiency
   - Uses experience replay for stable training
   - Implements epsilon-greedy exploration

## Results

Our model achieves strong performance metrics:

- Overall Accuracy: 88.82%
- Compute Savings: 38.8%
- Effectiveness Score: 34.46

Exit Point Distribution:
- Exit 1: 5.3%
- Exit 2: 34.3%
- Exit 3: 31.7%
- Exit 4: 28.6%

Per-Class Performance:
- Best Classes: Car (96.0%), Frog (94.3%), Ship (93.7%)
- Most Challenging: Dog (79.3%), Cat (82.1%), Bird (83.2%)

## Project Structure

```
src/
├── models/               # Model architectures
│   ├── early_exit_cnn.py  # CNN implementation
│   ├── dqn_agent.py       # DQN agent
│   └── environment.py     # Training environment
├── training/            # Training implementations
│   ├── train_cnn.py      # CNN training
│   └── train_rl.py       # RL training
├── evaluation/         # Evaluation code
│   └── evaluate.py     # Evaluation metrics
├── inference/         # Inference implementation
│   └── inference.py   # Inference code
└── visualization/    # Visualization tools
    └── visualize.py  # Plotting functions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Shikha-code36/early-exit-cnn.git
cd early-exit-cnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Train the CNN model:
```python
from src.training.train_cnn import pretrain_cnn
from src.data.data_loader import load_cifar10_data

# Load data
train_loader, test_loader = load_cifar10_data(batch_size=128)

# Train model
losses, accuracies = pretrain_cnn(model, train_loader, num_epochs=50)
```

2. Train the RL agent:
```python
from src.training.train_rl import train_rl_agent

rewards, exit_counts = train_rl_agent(
    model,
    agent,
    env,
    train_loader,
    num_episodes=5000
)
```

### Inference

Run inference on new images:
```python
from src.inference.inference import EarlyExitInference

# Initialize inference
inferencer = EarlyExitInference(model_path='models/')

# Process image
result = inferencer.process_image("path/to/image.jpg")
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Exit Point: {result['exit_point']}")
```

### Evaluation

Evaluate model performance:
```python
from src.evaluation.evaluate import evaluate_model

metrics = evaluate_model(model, agent, test_loader)
print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Compute Saved: {metrics['compute_saved']:.2f}%")
```

## Model Architecture Details

The Early Exit CNN employs a progressive architecture:

1. **First Exit (64 channels)**:
   - Basic feature extraction
   - Early exit for simple cases
   - 68.4% accuracy for easy classes

2. **Second Exit (128 channels)**:
   - Intermediate processing
   - Improved feature representation
   - 86.8% accuracy for moderate cases

3. **Third Exit (256 channels)**:
   - Advanced feature processing
   - Enhanced classification capability
   - 92.9% accuracy for complex cases

4. **Final Exit (512 channels)**:
   - Deep feature extraction 
   - Comprehensive classification
   - 92.2% accuracy for challenging cases

## Visualizations

The project includes several visualization tools:

1. Training Progress:
   - Loss curves
   - Accuracy per exit point
   - Exit distribution

2. Analysis Tools:
   - Confidence distributions
   - Class-wise exit patterns
   - Performance heatmaps

Example visualization code:
```python
from src.visualization.visualize import plot_training_metrics

plot_training_metrics(
    train_losses=losses,
    accuracies_per_exit=accuracies,
    exit_distributions=exit_dist
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{early-exit-cnn-2024,
  author = {Shikha Pandey},
  title = {Early Exit CNN with RL-based Decision Making},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Shikha-code36/early-exit-cnn}
}
```

## Acknowledgments

- CIFAR-10 dataset
- PyTorch team for the deep learning framework
- Reinforcement learning community for DQN implementations