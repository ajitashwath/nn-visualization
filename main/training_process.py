import tensorflow as tf
import matplotlib.pyplot as plt

class TrainingProcess(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs = None):
        self.losses.append(logs['loss'])
        self.accuracies.append(logs['accuracy'])
        self.plot_progress()
    def plot_progress(self):
        plt.figure(figsize = (12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label = 'Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies, label = 'Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shapes = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    X = tf.random.normal((1000, 784))
    y = tf.random.normal((1000, ), maxval = 10, dtype = tf.int32)
    visualizer = TrainingProcess()
    model.fit(X, y, epochs = 10, callbacks = [visualizer])
