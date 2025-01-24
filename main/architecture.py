import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def visualize_arch(model, file_name = "model_arch.png"):
    plot_model(model, to_file = file_name, show_shapes = True, show_layer_names = True)
    print(f"Model architecture saved to {file_name}")

if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shape = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    visualize_arch(model)