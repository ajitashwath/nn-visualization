import streamlit as st 
import tensorflow as tf
import numpy as np
from main.architecture import visualize_arch
from main.activations import visualize_act
from main.decision_boundaries import plot_decision_boundary
from main.gradients import visualize_grad
from main.training_process import TrainingProcess
from sklearn.datasets import make_moons

st.title("Neural Network Visualization Toolkit")

st.title('''
         Welcome to the Neural Network Visualization Toolkit
         This app helps you visualize various aspects of neural
         network, including architecture, activations, gradients and more.
         ''')

st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose a visualization", ["Architecture", "Activations", "Decision Boundaries", 'Gradients', 'Training Progress'])

st.sidebar.header("Hyperparameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, 16)
epochs = st.sidebar.slider.("Epochs", 1, 100, 10, 1)

model_type = st.sidebar.selectbox("Model Type", ["Simple Neural Network", "Convolutional Neural Network"])

st.sidebar.header("Custom Data")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type = ["csv"])

if option == "Architecture":
    st.header("Visualize Network Architecture")
    if model_type == 'Simple Neural Network':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation = 'relu', input_shape = (784, )),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(10, activation ='softmax')
        ])
    elif model_type == 'Convolutional Neural Network':
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shapes = (28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(64, activation = 'softmax')
        ])
    else:
        st.warning("Please select a valid model type")
    visualize_arch(model)

elif option == "Activations":
    st.header("Visualize Activation Maps")
    if model_type == "Simple Neural Network":
        st.warning("This visualization is only available for Convolutional Neural Networks")
    elif model_type == 'Convolutional Neural Network':
        model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False)
        input_data = tf.random.normal((1, 224, 224, 3))
        visualize_act(model, input_data, 'block1_conv1')
    else:
        st.warning("Please select a valid activation type")

elif option == 'Decision Boundaries':
    st.header("Visualize Decision Boundaries")
    X, y = make_moons(n_samples = 1000, noise = 0.2, random_state = 42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation = 'relu', input_shape = (2, )),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X, y, epochs = 100, verbose = 0)
    plot_decision_boundary(model, X, y)

elif option == "Gradients":
    st.header("Visualize Gradients")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shapes = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    X = tf.random.normal((batch_size, 784))
    y = tf.random.uniform((batch_size, ), maxval = 10, dtype = tf.int32)
    visualize_grad(model, X, y)

elif option == "Training Progress":
    st.header("Visualize Training Progress")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shapes = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    X = tf.random.normal((1000, 784))
    y = tf.random.normal((1000, ), maxval = 10, dtype = tf.int32)
    visualizer = TrainingProcess()
    model.fit(X, y, epochs = epochs, batch_size = batch_size, callbacks = [visualizer])

st.sidebar.write("Built by me.")