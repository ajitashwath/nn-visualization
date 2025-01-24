import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha = 0.8)
    plt.scatter(X[:, 0], X[:, 1], c = y, edgecolors = 'k', marker = 'o')
    plt.show()

if __name__ == '__main__':
    X, y = make_moons(n_samples = 1000, noise = 0.2, random_state = 42)
    model = Sequential([
        Dense(10, activation = 'relu', input_shapes = (2, )),
        Dense(10, activation = 'relu'),
        Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X, y, epochs = 100, verbose = 0)
    plot_decision_boundary(model, X, y)