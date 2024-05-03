import numpy as np
from math import sqrt
from numpy import asarray
from numpy.random import rand, seed
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from space_3d import objective
from space_2d import plot2d


def derivative(x, y):
    """Gradient of the convex objective function: z' = 2x + 2y"""
    return asarray([x * 2.0, y * 2.0])

def lossMSE(x, y_true):
    """Calculates the Euclidean distance between the current point and the target point."""
    return np.sqrt((x[0] - y_true[0])**2 + (x[1] - y_true[1])**2)


def GD(objective, derivative, bounds, n_iter, alpha, eps=1e-8):
    """Gradient Descent algorithm"""
    solutions = []

    # Generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    y_true = np.array([0, 0])
    # Update the gradient descent
    for t in range(n_iter):
        # Calculate the gradient
        g = derivative(x[0], x[1])

        # Construct the solution one variable at a time
        for i in range(bounds.shape[0]):

            # Update the solution
            x[i] = x[i] - alpha * g[i] 

        # Evaluate the candidate point
        score = objective(x[0], x[1])

        loss_value = lossMSE(x, y_true)

        # Keep track of the solutions
        solutions.append(x.copy())

        print(f'>{t} f({str(x)}) = {score:.5f}, MSE = {loss_value:.5f}')

    return solutions


if __name__ == '__main__':
    seed(2)  # Set the random seed
    bounds = asarray([[-1., 2.], [-1., 1.]])
    n_iter = 300  # Total iterations
    alpha = 0.02  # Step size

    solutions = GD(objective, derivative, bounds, n_iter, alpha)

    # Plot the solutions
    plot2d(bounds, objective, solutions)
    


