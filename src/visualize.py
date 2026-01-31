import numpy as np  # Import NumPy for meshgrid and numeric arrays
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from .model import predict_linear  # Import prediction function
from .metrics import mse  # Import MSE function


def plot_data(x, y, title="Data"):  # Define a helper to plot the dataset
    plt.figure(figsize=(7, 4))  # Create a figure with a consistent size
    plt.scatter(x, y, label="data")  # Plot points as scatter
    plt.title(title)  # Add plot title
    plt.xlabel("x")  # Label x-axis
    plt.ylabel("y")  # Label y-axis
    plt.grid(True)  # Add grid for easier reading
    plt.legend()  # Show legend
    plt.show()  # Display the plot


def plot_predictions_for_ws(
    x, y, w_list, b_fixed=0.0
):  # Define plot showing predictions for multiple weights
    plt.figure(figsize=(8, 5))  # Create the figure
    plt.scatter(x, y, label="data")  # Plot the real data

    for w in w_list:  # Loop through each weight value
        y_hat = predict_linear(x, w, b_fixed)  # Predict using current w and fixed b
        current_mse = mse(y, y_hat)  # Compute MSE for this line
        plt.plot(
            x, y_hat, label=f"w={w}, MSE={current_mse:.2f}"
        )  # Plot line and show MSE in legend

    plt.title("Predictions for different weights (b fixed)")  # Add title
    plt.xlabel("x")  # Label x-axis
    plt.ylabel("y / y_hat")  # Label y-axis
    plt.grid(True)  # Add grid
    plt.legend()  # Show legend
    plt.show()  # Display plot


def compute_mse_vs_w(
    x, y, w_values, b_fixed=0.0
):  # Define a function that computes MSE for many weights
    mse_values = []  # Create empty list for errors
    for w in w_values:  # Loop through weights
        y_hat = predict_linear(x, w, b_fixed)  # Predict using current w
        mse_values.append(mse(y, y_hat))  # Compute MSE and store it
    return np.array(mse_values)  # Return errors as NumPy array


def plot_mse_vs_w(
    w_values, mse_values, title="MSE vs w"
):  # Define function to plot error curve
    plt.figure(figsize=(7, 4))  # Create figure
    plt.plot(w_values, mse_values)  # Plot MSE values against weights
    plt.title(title)  # Add title
    plt.xlabel("w")  # Label x-axis
    plt.ylabel("MSE")  # Label y-axis
    plt.grid(True)  # Add grid
    plt.show()  # Display plot


def compute_mse_surface(
    x, y, w_grid, b_grid
):  # Define function to compute MSE over a (w,b) grid
    W, B = np.meshgrid(w_grid, b_grid)  # Create grid matrices for w and b
    Z = np.zeros_like(W, dtype=float)  # Create matrix to store MSE values

    for i in range(B.shape[0]):  # Loop over rows (b dimension)
        for j in range(W.shape[1]):  # Loop over columns (w dimension)
            w = float(W[i, j])  # Read current w from grid
            b = float(B[i, j])  # Read current b from grid
            y_hat = predict_linear(x, w, b)  # Predict using grid parameters
            Z[i, j] = mse(y, y_hat)  # Store the computed MSE in Z

    return W, B, Z  # Return the full surface (W,B,Z)


def plot_mse_surface_contour(
    W, B, Z, title="MSE surface (contour)"
):  # Define contour plot for MSE surface
    plt.figure(figsize=(8, 5))  # Create figure
    contour = plt.contourf(W, B, Z, levels=30)  # Create filled contour plot
    plt.colorbar(contour)  # Add colorbar to show MSE scale
    plt.title(title)  # Add title
    plt.xlabel("w")  # Label w-axis
    plt.ylabel("b")  # Label b-axis
    plt.grid(True)  # Add grid
    plt.show()  # Display plot


def plot_mse_history(
    history_mse, title="MSE over steps"
):  # Define plot for training curve
    plt.figure(figsize=(7, 4))  # Create figure
    plt.plot(history_mse)  # Plot MSE values across iterations
    plt.title(title)  # Add title
    plt.xlabel("Step")  # Label x-axis
    plt.ylabel("MSE")  # Label y-axis
    plt.grid(True)  # Add grid
    plt.show()  # Display plot


def plot_gd_path_on_surface(
    W, B, Z, history_w, history_b, title="Gradient descent path"
):  # Define overlay plot for GD path
    plt.figure(figsize=(8, 5))  # Create figure
    contour = plt.contourf(W, B, Z, levels=30)  # Plot MSE contours
    plt.colorbar(contour)  # Add colorbar
    plt.plot(history_w, history_b, marker="o")  # Overlay GD path points
    plt.title(title)  # Add title
    plt.xlabel("w")  # Label x-axis
    plt.ylabel("b")  # Label y-axis
    plt.grid(True)  # Add grid
    plt.show()  # Display plot
