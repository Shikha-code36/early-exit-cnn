import torch
import torch.nn.functional as F


class EarlyExitEnvironment:
    def __init__(self, model, compute_penalty=0.05):  # Reduced penalty
        self.model = model
        self.compute_penalty = compute_penalty
        self.current_exits = None
        self.current_target = None
        self.exit_index = 0
        # Adjusted confidence thresholds
        self.confidence_thresholds = [0.70, 0.80, 0.85, 0.0]

    def reset(self, input_data, target):
        with torch.no_grad():
            self.current_exits = self.model(input_data, return_all_exits=True)
            self.current_target = target
            self.exit_index = 0
    
            confidences = F.softmax(self.current_exits[0], dim=1)
            max_conf = confidences.max(1)[0]
            return max_conf.cpu().numpy() 

    def step(self, action):
        done = False
        reward = 0
    
        # Calculate confidence scores first
        confidences = F.softmax(self.current_exits[self.exit_index], dim=1)
        max_conf = confidences.max(1)[0]
    
        if action == 1:  # continue
            if self.exit_index < len(self.current_exits) - 1:
                self.exit_index += 1
                # Smaller penalty for continuing
                reward = -self.compute_penalty
            else:
                done = True
    
            next_state = max_conf.cpu().numpy()  # Added .cpu()
    
        else:  # exit
            prediction = self.current_exits[self.exit_index].argmax(1)
            correct = (prediction == self.current_target).item()
    
            # Enhanced rewards for correct predictions at later exits
            if correct:
                base_reward = 1.0
                # Higher reward for correct early exits
                exit_bonus = (3 - self.exit_index) * 0.3
                confidence_bonus = max_conf.item() * 0.2
                reward = base_reward + exit_bonus + confidence_bonus
            else:
                # Larger penalty for incorrect early exits
                confidence_penalty = (1 - max_conf.item()) * 0.5
                reward = -1.0 - confidence_penalty
    
            done = True
            next_state = None
    
        return next_state, reward, done