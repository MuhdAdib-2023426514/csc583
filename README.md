# PyTorch CNN Image Classification Project

## Overview
This project is a PyTorch-based Convolutional Neural Network (CNN) for image classification. It uses the Intel Image Dataset to classify images into one of five classes:

- Buildings
- Street
- Mountain
- Forests
- Sea

The model is designed to leverage CUDA for GPU acceleration, ensuring efficient training and inference.

## Dataset
The Intel Image Dataset is used for training and testing the model. The dataset is organized into the following directories:

```
Data/
    buildings/
    forest/
    mountain/
    sea/
    street/
```

Each directory contains images corresponding to its respective class.

## Requirements
To run this project, ensure you have the following installed:

- Python 3.x
- PyTorch (with CUDA support)
- pandas
- Other dependencies listed in `pyproject.toml`

## Files
- `main.py`: The main script for training and evaluating the model.
- `intel_cnn_model.pth`: The saved PyTorch model.
- `training_history.csv` and `training_history1.csv`: Files containing training history logs.
- `output.txt`: Output logs from the training process.
- `intelImage_classification.ipynb`: Jupyter Notebook for interactive experimentation.

## Usage

### 1. Setup
Ensure you have a CUDA-enabled GPU and the required dependencies installed. You can install the dependencies using:

```
pip install -r requirements.txt
```

Alternatively, if you are using `pyproject.toml`, use:

```
poetry install
```

### 2. Training the Model
Run the `main.py` script to train the model:

```
python main.py
```

### 3. Evaluating the Model
After training, the model can be evaluated using the test dataset. Modify the `main.py` script or use the Jupyter Notebook for evaluation.

### 4. Inference
Use the trained model (`intel_cnn_model.pth`) to classify new images. Load the model in your script or notebook and pass the image for prediction.

## Results
The training history is logged in `training_history.csv` and `training_history1.csv`. These files contain metrics such as accuracy and loss over epochs.

## Acknowledgments
- Intel Image Dataset: [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- PyTorch: [PyTorch Official Website](https://pytorch.org/)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Setting Up the Project with `uv`

This project uses `uv` for managing dependencies and environments. Follow the steps below to set up the project:

### 1. Install `uv`
Ensure `uv` is installed on your system. If not, install it using:

```
pip install uv
```

### 2. Add Dependencies
To add dependencies, use the `uv add` command. For example, to add `pandas`:

```
uv add pandas
```

### 3. Install Dependencies
Install all dependencies listed in the `pyproject.toml` file by running:

```
uv install
```

### 4. Run the Project
Once the dependencies are installed, you can run the project scripts. For example, to train the model:

```
python main.py
```

### 5. Update Dependencies
To update dependencies, use:

```
uv update
```

For more information on `uv`, refer to the [official documentation](https://uv-py.readthedocs.io/).