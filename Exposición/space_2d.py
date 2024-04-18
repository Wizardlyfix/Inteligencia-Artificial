import numpy as np
from numpy import asarray, arange, meshgrid
from matplotlib import pyplot
from space_3d import objective


def plot2d(bounds: np.array, function, solutions: list = list()):
    '''Given an array defining the space and a function to use, plot the 2D space.'''

    # Sample the space defined by the accepted range.
    xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
    yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)

    # Create a grid.
    x, y = meshgrid(xaxis, yaxis)

    # Apply the function to the grid.
    results = function(x, y)

    # Plot the function on a 2D graph, with 50 levels and a jet color map.
    pyplot.contourf(x, y, results, levels=50, cmap='jet')

    # Plot any solutions provided.
    if solutions:
        solutions = asarray(solutions)
        pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')

    # Display the plot.
    pyplot.show()


if __name__ == '__main__':
    # Define the bounds.
    bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
    print("Bounds are: ", bounds)

    # Plot the 2D space.
    plot2d(bounds, objective)
