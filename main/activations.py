import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_act(model, input_data, layer_name):
    intermediate_model = tf.keras.Model(inputs = model.input, outputs = model.get_layer(layer_name).output)
    activations = intermediate_model.predict(input_data)

    plt.figure(figsize = (10, 10))
    for i in range(min(activations.shape[-1], 16)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(activations[0, :, :, i], cmap = 'viridis')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False)
    input_data = tf.random.normal((1, 224, 224, 3))
    visualize_act(model, input_data, 'block1_conv1')