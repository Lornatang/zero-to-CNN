import random
import math
import numpy as np
import datetime
import Activators


# fully connected implementation classes for each layer
class FullConnected(object):
    """
      input_size: input vector.
      output_size: output vector.
      activator: Tanh, Sigmoid, ReLU.
    """

    def __init__(self, input_size, output_size, activator, learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # rand(output_size, input_size) - 0.5) * 2 * math.sqrt(6 / (output_size + input_size))
        wimin = (output_size - 0.5) * 2 * \
            math.sqrt(6 / (input_size + output_size))
        wimax = (input_size-0.5)*2*math.sqrt(6/(input_size + output_size))
        # Initializes to a number between -0.1 and 0.1. The size of the weights.
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # bias vector
        self.b = np.zeros((output_size, 1))
        # learning rate
        self.learning_rate = learning_rate
        # output vector
        self.output = np.zeros((output_size, 1))

    # Forward calculation, prediction of output.
    def forward(self, x):
        self.input = x
        self.output = self.activator.forward(
            np.dot(self.W, x) + self.b)

    # Reverse the gradient of W and b.
    def backward(self, delta_array):
        self.delta = np.multiply(self.activator.backward(
            self.input), np.dot(self.W.T, delta_array))
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    # Update the weights using gradient descent.
    def update(self):
        self.W += self.learning_rate * self.W_grad
        self.b += self.learning_rate * self.b_grad
