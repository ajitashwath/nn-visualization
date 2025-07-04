# Neural Network Visualization Toolkit
A web application for visualizing various aspects of neural networks, including architecture, activations, decision boundaries, gradients, and training progress.

## Features
- **Architecture Visualization**: Display neural network structure with layer details
- **Activation Maps**: Visualize feature maps in convolutional layers
- **Decision Boundaries**: Plot classification boundaries for 2D datasets
- **Gradient Analysis**: Examine gradient distributions across layers
- **Training Progress**: Real-time monitoring of loss and accuracy during training

## Installation
1. Clone the repository:
```bash
git clone https://github.com/ajitashwath/nn-visualization.git
cd nn-visualization
```

2. Install required dependencies:
```bash
pip install streamlit tensorflow numpy matplotlib scikit-learn keras
```

## Usage

Launch the application:
```bash
streamlit run app.py
```

## Application Interface

### Sidebar Controls

- **Visualization Type**: Select from 5 different visualization options
- **Hyperparameters**: 
  - Learning Rate: 0.001 - 0.1
  - Batch Size: 16 - 128
  - Epochs: 1 - 100
- **Model Type**: Choose between Simple Neural Network or CNN
- **Custom Data**: Upload CSV datasets for analysis

### Visualization Options

#### 1. Architecture
Displays network structure using Keras plot_model functionality.

**Models Available:**
- Simple Neural Network: 784 → 64 → 64 → 10 (ReLU, Softmax)
- CNN: Conv2D(32) → MaxPool → Flatten → Dense(64) → Dense(10)

#### 2. Activations
Visualizes activation maps for convolutional layers.

**Features:**
- Uses pre-trained VGG16 model
- Displays up to 16 feature maps in 4x4 grid
- Only available for CNN models

#### 3. Decision Boundaries
Plots classification boundaries for 2D datasets.

**Implementation:**
- Uses make_moons dataset (1000 samples, 0.2 noise)
- Binary classification with sigmoid activation
- Visualizes decision contours with data points

#### 4. Gradients
Analyzes gradient distributions across network layers.

**Visualization:**
- Histogram of gradient values per layer
- Helps identify vanishing/exploding gradient problems
- Color-coded by layer

#### 5. Training Progress
Real-time monitoring of training metrics.

**Features:**
- Live plots of loss and accuracy
- Updates after each epoch
- Dual subplot layout

## File Structure (Main)

```
neural-network-visualization-toolkit/
├── app.py                      # Main Streamlit application
├── main/
│   ├── __init__.py
│   ├── architecture.py         # Network architecture visualization
│   ├── activations.py          # Activation map visualization
│   ├── decision_boundaries.py  # Decision boundary plotting
│   ├── gradients.py            # Gradient analysis
│   └── training_process.py     # Training progress callback
└── README.md
```

## Module Details

### app.py
Main application file containing:
- Streamlit UI configuration
- Model definitions
- Data preparation
- Visualization routing

### main/architecture.py
- **Function**: `visualize_arch(model)`
- **Purpose**: Generates and displays network architecture diagrams
- **Output**: PNG image of model structure

### main/activations.py
- **Function**: `visualize_act(model, input_data, layer_name)`
- **Purpose**: Extracts and visualizes activation maps
- **Parameters**: 
  - `model`: Keras model
  - `input_data`: Input tensor
  - `layer_name`: Target layer for visualization

### main/decision_boundaries.py
- **Function**: `plot_decision_boundary(model, X, y)`
- **Purpose**: Plots classification decision boundaries
- **Method**: Meshgrid prediction with contour plotting

### main/gradients.py
- **Function**: `visualize_grad(model, X, y)`
- **Purpose**: Analyzes gradient distributions
- **Method**: GradientTape for automatic differentiation

### main/training_process.py
- **Class**: `TrainingProcess(Callback)`
- **Purpose**: Real-time training visualization
- **Methods**:
  - `on_epoch_end()`: Updates plots after each epoch
  - `plot_progress()`: Generates loss/accuracy plots

## Technical Requirements
- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Keras

## Sample Datasets
The application includes built-in datasets:
- **MNIST-like**: 784-dimensional random data for simple networks
- **Make Moons**: 2D classification dataset for decision boundaries
- **ImageNet**: Pre-trained VGG16 for activation visualization
