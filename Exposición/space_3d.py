import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def objective(x: float, y: float) -> float:
    """
    Calculates the objective function: z = x**2 + y**2

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the objective function.
    """
    return x ** 2 + y ** 2


if __name__ == '__main__':
    # Range of input
    r_min, r_max = -1.0, 1.0

    # Sample the range uniformly
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)

    # Create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis, sparse=False)

    # Calculate the objective function
    results = objective(x, y)

    # Create a plot of the surface with the color scheme 'jet'
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, results, cmap="jet")

    # Display the plot
    plt.show()