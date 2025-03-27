import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_decision_boundary(model, X, y):
    try:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 
                           np.arange(y_min, y_max, 0.01))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
        ax.set_title('Decision Boundary')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        fig.colorbar(contour)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting decision boundary: {str(e)}")