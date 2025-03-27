import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st

def visualize_act(model, input_data, layer_name):
    try:
        intermediate_model = tf.keras.Model(inputs=model.input, 
                                         outputs=model.get_layer(layer_name).output)
        activations = intermediate_model.predict(input_data, verbose=0)

        fig = plt.figure(figsize=(10, 10))
        for i in range(min(activations.shape[-1], 16)):
            plt.subplot(4, 4, i + 1)
            plt.imshow(activations[0, :, :, i], cmap='viridis')
            plt.axis('off')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error visualizing activations: {str(e)}")