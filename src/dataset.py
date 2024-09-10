import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define transformations for the dataset (grayscale, resizing, normalization)
transform = transforms.Compose([
    transforms.Grayscale(),        # Convert to grayscale
    transforms.Resize((64, 64)),    # Resize to 64x64
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the image
])

# Define a custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if "dataset_custom" in root_dir:
            if not os.path.isdir(root_dir):
                raise ValueError("Custom dataset folder does not exist. Please provide the correct path.")
            print("Using custom dataset.")
        # Check if the folder dataset exists
        elif not os.path.isdir(root_dir):
            print("The dataset folder does not exist. Pulling the dataset from Kaggle...")
            try:
                os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')
            except:
                # Throw an error if the Kaggle API key is not found
                raise ValueError("Kaggle API key not found. Please follow the instructions to set up the Kaggle API key.")

            dataset = "jonathanoheix/face-expression-recognition-dataset"
            os.system(f'kaggle datasets download -d {dataset} -p ./dataset --unzip')
            print("Dataset downloaded successfully.")

        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # Assumes folders are labeled with classes
        self.image_paths = []
        self.labels = []
        
        # Loop through each class folder and store image paths and their labels
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # Apply transformation if any
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label
    
if __name__ == "__main__":
    # Instantiate dataset for training and validation
    train_dataset = EmotionDataset(root_dir='myNetwork/dataset/images/train', transform=transform)
    validation_dataset = EmotionDataset(root_dir='myNetwork/dataset/images/validation', transform=transform)

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    # Check the classes (should be ['neutral', 'happy', 'sad', 'angry', 'surprise', 'disgust', 'fear'])
    print(train_dataset.classes)
