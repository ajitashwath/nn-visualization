import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st

def visualize_grad(model, X, y):
    try:
        with tf.GradientTape() as tape:
            predictions = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)

        fig = plt.figure(figsize=(10, 6))
        for i, grad in enumerate(gradients):
            if grad is not None:
                plt.hist(grad.numpy().flatten(), bins=50, alpha=0.7, label=f'Layer {i + 1}')
        plt.xlabel('Gradient Value')
        plt.ylabel('Frequency')
        plt.title('Gradient Distribution')
        plt.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error visualizing gradients: {str(e)}")