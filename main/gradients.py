import tensorflow as tf
import matplotlib.pyplot as plt

def visualize_grad(model, X, y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    plt.figure(figsize = (10, 6))
    for i, grad in enumerate(gradients):
        if grad is not None:
            plt.hist(grad.numpy().flatten(), bins = 50, alpha = 0.7, label = f'Layer {i + 1}')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.title('Gradient Distribution')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shape = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    X = tf.random.normal((32, 784))
    y = tf.random.uniform((32, ), maxval = 10, dtype = tf.int32)
    visualize_grad(model, X, y)