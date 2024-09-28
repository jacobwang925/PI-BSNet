import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def plot_all_frames(data):
    # Ensure the data has 4 dimensions
    assert data.ndim == 4, "Data must be a 4D numpy array"

    # Extract the dimensions
    time_steps, x_dim, y_dim, z_dim = data.shape

    # Calculate global min and max
    global_min = np.min(data)
    global_max = np.max(data)

    # Determine the grid size
    cols = 10  # Number of columns in the grid
    rows = (time_steps + cols - 1) // cols  # Number of rows in the grid

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

    for frame in range(time_steps):
        row = frame // cols
        col = frame % cols
        ax = axes[row, col]

        # Plot the data for the current frame
        im = ax.matshow(data[frame, :, :, z_dim//2], cmap='viridis')#, vmin=global_min, vmax=global_max)

        # Set title
        ax.set_title(f'Frame {frame}')
        ax.axis('off')

    # Add a colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.show()