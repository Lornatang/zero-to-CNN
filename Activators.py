import numpy as np


# ReLU activator
class ReLU(object):
    def forward(self, weighted_input):    # forward calculation, calculation output
        return max(0, weighted_input)

    def backward(self, output):  # backward calculation, calculation derivative
        return 1 if output > 0 else 0


# Sigmoid activator
class Sigmoid(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return np.multiply(output, (1 - output))


# Tanh activator
class Tanh(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output
