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


def adam(objective, derivative, bounds, n_iter, alpha, eps=1e-8):
    """Adam algorithm"""
    solutions = []

    # Generate an initial point
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # Initialize the first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    # Update the gradient descent
    for t in range(n_iter):
        # Calculate the gradient
        g = derivative(x[0], x[1])

        # Construct the solution one variable at a time
        for i in range(bounds.shape[0]):
            # Update the first moment
            #m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]

            # Update the second moment
            #v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2

            # Compute the bias-corrected first and second moments
            #mhat = m[i] / (1 - beta1 ** (t + 1))
            #vhat = v[i] / (1 - beta2 ** (t + 1))

            # Update the solution
            x[i] = x[i] - alpha * g[i] # mhat / (sqrt(vhat) + eps)

        # Evaluate the candidate point
        score = objective(x[0], x[1])

        # Keep track of the solutions
        solutions.append(x.copy())

        print(f'>{t} f({str(x)}) = {score:.5f}')

    return solutions


if __name__ == '__main__':
    seed(2)  # Set the random seed
    bounds = asarray([[-1., 2.], [-1., 1.]])
    n_iter = 600  # Total iterations
    alpha = 0.02  # Step size
    #beta1 = 0.8  # Gradient mean factor
    #beta2 = 0.999  # Gradient square mean factor

    solutions = adam(objective, derivative, bounds, n_iter, alpha) #, beta1, beta2)

    # Plot the solutions
    plot2d(bounds, objective, solutions)
    


