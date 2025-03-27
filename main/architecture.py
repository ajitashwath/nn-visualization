import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import streamlit as st

def visualize_arch(model):
    try:
        fig = tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
        st.image('model.png', caption='Neural Network Architecture')
    except Exception as e:
        st.error(f"Error visualizing architecture: {str(e)}")

if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shape = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    visualize_arch(model)