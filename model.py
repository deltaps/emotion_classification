import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # 1st convolutional layer: 1 input channel (grayscale image), 16 output channels, kernel size 3x3
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming input images are 64x64
        self.fc2 = nn.Linear(128, 7)  # 7 output classes for emotions
        
        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Apply convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the image before the fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax needed here, we'll use CrossEntropyLoss later
        
        return x

if __name__ == "__main__":
    # Instantiate the model
    model = EmotionCNN()
    print(model)
