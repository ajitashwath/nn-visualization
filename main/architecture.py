import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import streamlit as st

def visualize_arch(model):
    try:
        plot_model(model, show_shapes = True, show_layer_names = True, to_file = 'model.png')
        st.image('model.png', caption='Neural Network Architecture')
    except Exception as e:
        st.error(f"Error visualizing architecture: {str(e)}")

if __name__ == "__main__":
    model = Sequential([
        Dense(64, activation = 'relu', input_shape = (784, )),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax')
    ])
    visualize_arch(model)