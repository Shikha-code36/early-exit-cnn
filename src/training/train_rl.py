import torch
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_rl_agent(model, agent, env, train_loader, num_episodes=1000):
    """
    Train the RL agent to learn when to exit the network.
    Modified for CIFAR-10 with adjusted rewards and monitoring.
    """
    print("Training RL agent...")
    model.eval()  # Set CNN to evaluation mode

    episode = 0
    total_rewards = []
    exit_counts = [0, 0, 0, 0]  # Track which exits are being used
    running_accuracy = []

    progress_bar = tqdm(total=num_episodes, desc='Training RL agent')

    while episode < num_episodes:
        for data, target in train_loader:
            if episode >= num_episodes:
                break

            data, target = data.to(device), target.to(device)

            for i in range(len(data)):  # Process each sample in batch
                # Reset environment with new sample
                state = env.reset(data[i:i+1], target[i:i+1])
                total_reward = 0
                correct = False

                while True:
                    # Get action from agent
                    action = agent.get_action(state)

                    # Take action in environment
                    next_state, reward, done = env.step(action)

                    if done:
                        exit_counts[env.exit_index] += 1
                        if reward > 0:  # Track if prediction was correct
                            correct = True

                    # Store experience in memory
                    if next_state is not None:
                        agent.remember(state, action, reward, next_state, done)

                    total_reward += reward

                    if done:
                        break

                    state = next_state

                # Train agent
                agent.train()

                total_rewards.append(total_reward)
                running_accuracy.append(1.0 if correct else 0.0)
                if len(running_accuracy) > 100:  # Keep track of last 100 episodes
                    running_accuracy.pop(0)

                episode += 1

                # Update progress
                if episode % 10 == 0:
                    avg_reward = sum(total_rewards[-100:]) / min(len(total_rewards), 100)
                    avg_accuracy = sum(running_accuracy) / len(running_accuracy)
                    exit_distribution = [count/episode*100 for count in exit_counts]

                    progress_bar.set_postfix({
                        'avg_reward': f'{avg_reward:.2f}',
                        'accuracy': f'{avg_accuracy:.2f}',
                        'epsilon': f'{agent.epsilon:.2f}',
                        'exit_dist': f'[{exit_distribution[0]:.1f}%, {exit_distribution[1]:.1f}%, {exit_distribution[2]:.1f}%, {exit_distribution[3]:.1f}%]'
                    })
                    progress_bar.update(10)

                if episode >= num_episodes:
                    break

    print("RL agent training completed!")
    return total_rewards, exit_counts