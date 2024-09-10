import torch.optim as optim
from model import EmotionCNN
from dataset import EmotionDataset, transform
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch

# Check if CUDA is available, and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Instantiate the model
model = EmotionCNN().to(device)

# Instantiate dataset for training and validation
train_dataset = EmotionDataset(root_dir='dataset/images/train', transform=transform)
validation_dataset = EmotionDataset(root_dir='dataset/images/validation', transform=transform)

# Create DataLoader for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting Training...")
# Training loop
for epoch in range(10):  # Loop over the dataset 10 times
    running_loss = 0.0
    for images, labels in tqdm(train_loader):

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")

print("Finished Training")

# Save the model
torch.save(model.state_dict(), 'save_weight/model.pth')