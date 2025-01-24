import streamlit as st 
import tensorflow as tf
from main.architecture import visualize_arch
from main.activations import visualize_act
from main.decision_boundaries import plot_decision_boundary

st.title("Neural Network Visualization Toolkit")

st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose a visualization", ["Architecture", "Activations", "Decision Boundaries"])

if option == "Architecture":
    st.header("Visualize Network Architecture")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu', input_shape = (784, )),
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dense(10, activation ='softmax')
    ])
    visualize_arch(model)

elif option == "Activations":
    st.header("Visualize Activation Maps")
    model = tf.keras.applications.VGG16(weights = 'imagenet', include_top = False)
    input_data = tf.random.normal((1, 224, 224, 3))
    visualize_act(model, input_data, 'block1_conv1')

elif option == 'Decision Boundaries':
    st.header("Visualize Decision Boundaries")
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples = 1000, noise = 0.2, random_state = 42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation = 'relu', input_shape = (2, )),
        tf.keras.layers.Dense(10, activation = 'relu'),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(X, y, epochs = 100, verbose = 0)
    plot_decision_boundary(model, X, y)
