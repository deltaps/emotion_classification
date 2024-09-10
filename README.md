# Emotion Classification Neural Network

This project is focused on creating a neural network that can classify emotions based on facial images. The model is trained on a dataset containing images of faces with different expressions, allowing it to learn and recognize emotions like happiness, sadness, anger, etc.

## Project Structure

- **dataset/**: Contains the dataset used for training and validation.
- **checkpoints/**: Folder to store model weights.
- **src/**: Contains the main code files for dataset processing, model definition, training, and testing.
    - `dataset.py`: Handles data loading and preprocessing.
    - `model.py`: Defines the neural network architecture.
    - `training.py`: Script for training the model.
    - `test_stream_webcam.py`: Script to test the model using your webcam.
- **README.md**: Documentation for the project.

## Installation

### Prerequisites

- Python 3.12

### Setup

1. **Clone the repository** and navigate to the project folder: 
    ```bash
    git clone <repository-url> 
    cd <project-folder>
    ```
1. **Create a Python virtual environment**:
    ```bash
    python -m venv .venv
    ```
1. **Activate the virtual environment**:
    - On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    - On Windows:
        ```cmd
        .venv\Scripts\activate
        ```
1. **Install Poetry**:
    ```bash
    pip install poetry
    ```
1. **Install all dependencies**:
    ```bash
    poetry install
    ```
### Kaggle Dataset API Key Setup

To download datasets from Kaggle, you'll need to create a folder and add your Kaggle API key:

1. **Create a folder named** `.kaggle` in your home directory:
    - On macOS/Linux:
        ```bash
        mkdir ~/.kaggle
        ```
    - On Windows (cmd):
        ```cmd
        mkdir %USERPROFILE%\.kaggle
        ```
1. **Download the** `kaggle.json` API key file from your Kaggle account and place it in the .kaggle folder. The file is used for Kaggle dataset access.

## Running the Training

To start training the neural network, run the `training.py` script:

```bash
python src/training.py
```

This script will load the dataset, process it, train the model, and save the trained model weights to the `checkpoints/` folder.

## Testing the Model with Webcam

You can test the model using your computer's webcam to classify emotions in real-time. Run the `test_stream_webcam.py` script:

```bash
python src/test_stream_webcam.py
```

This script captures video from your webcam, processes each frame, and predicts the emotion being displayed.