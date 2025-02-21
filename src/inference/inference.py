import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your model architecture
from ee_cifar10_perclass import EarlyExitCNN, DQNAgent

class EarlyExitInference:
    def __init__(self, model_path='models/'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.cnn_model = EarlyExitCNN().to(self.device)
        self.dqn_agent = DQNAgent(state_size=1, action_size=2)
        
        # Load saved models
        self.load_models(model_path)
        
        # Set models to evaluation mode
        self.cnn_model.eval()
        self.dqn_agent.q_network.eval()
        
        # CIFAR-10 classes
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])

    def load_models(self, model_path):
        # Load CNN
        cnn_checkpoint = torch.load(f'{model_path}/early_exit_cnn.pth')
        self.cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
        
        # Load DQN
        dqn_checkpoint = torch.load(f'{model_path}/dqn_agent.pth')
        self.dqn_agent.q_network.load_state_dict(dqn_checkpoint['q_network_state_dict'])
        self.dqn_agent.epsilon = dqn_checkpoint['epsilon']

    def process_image(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get predictions from all exits
            exits = self.cnn_model(image_tensor, return_all_exits=True)
            
            # Initialize with first exit's confidence
            confidences = F.softmax(exits[0], dim=1)
            state = confidences.max(1)[0].cpu().numpy()
            exit_point = 0
            
            # Let agent decide when to exit
            while True:
                action = self.dqn_agent.get_action(state)
                
                if action == 0 or exit_point == len(exits) - 1:  # Exit or reached final layer
                    break
                    
                exit_point += 1
                confidences = F.softmax(exits[exit_point], dim=1)
                state = confidences.max(1)[0].cpu().numpy()
            
            # Get final prediction
            final_confidences = F.softmax(exits[exit_point], dim=1)
            pred_class = exits[exit_point].argmax(1).item()
            confidence = final_confidences.max(1)[0].item()
            
            return {
                'class': self.classes[pred_class],
                'confidence': confidence,
                'exit_point': exit_point + 1
            }

# Example usage
if __name__ == "__main__":
    # Initialize inference
    inferencer = EarlyExitInference()
    
    # Process a single image
    image_path = "path/to/your/test/image.jpg"
    result = inferencer.process_image(image_path)
    
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Exit point: {result['exit_point']}")

    # Process multiple images in a directory
    import os
    test_dir = "path/to/test/directory"
    
    for image_name in os.listdir(test_dir):
        if image_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(test_dir, image_name)
            result = inferencer.process_image(image_path)
            
            print(f"\nImage: {image_name}")
            print(f"Prediction: {result['class']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Exit point: {result['exit_point']}")