import numpy as np
import Activators
import math


# get conv region
def get_patch(input_channels, i, j, filter_width, filter_height, stride):
    """Take the region of the convolution from the input array,
       Automatic adaptation of input to 2D and 3D cases.

      input_channels: image channel matrix.
      i: lateral migration.
      j: vertical migration.
      filter_width: filter width.
      filter_height: filter height.
      stride: step length.
    """
    start_i = i * stride
    start_j = j * stride
    if input_channels.ndim == 2:  # GRAY
        return input_channels[start_i:start_i + filter_height, start_j: start_j + filter_width]
    elif input_channels.ndim == 3:  # RGB
        return input_channels[:, start_i: start_i + filter_height, start_j: start_j + filter_width]


# gets the index of the maximum value of a 2D region
def get_max_index(array):
    location = np.where(array == np.max(array))
    return location[0], location[1]


# computes the convolution of a filter and outputs a two-dimensional data.
def conv(input_channels, output_channels, kernel_size, stride, bias):
    output_width = output_channels.shape[1]
    output_height = output_channels.shape[0]
    # The width of the filter. There may be multiple channels. Shape =[depth, height, width] for multiple channels, shape=[height, width] for single channel.
    kernel_width = kernel_size.shape[-1]
    kernel_height = kernel_size.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            # get input conv region.
            juanjiqu = get_patch(
                input_channels, i, j, kernel_width, kernel_height, stride)
            # the convolution region convolves with the filter.
            kernel_values = (np.multiply(juanjiqu, kernel_size)).sum()
            # add the convolution result to the partial.
            output_channels[i][j] = kernel_values + bias


# zero padding.
def padding(input_channels, zp):
    if zp == 0:
        return input_channels
    else:
        if input_channels.ndim == 3:
            input_height = input_channels.shape[1]
            input_depth = input_channels.shape[0]
            input_width = input_channels.shape[2]
            in_channels = np.zeros(
                (input_depth, input_height + 2 * zp, input_width + 2 * zp))
            in_channels[:,
                        zp: zp + input_height,
                        zp: zp + input_width] = input_channels
            return in_channels
        elif input_channels.ndim == 2:
            input_height = input_channels.shape[0]
            input_width = input_channels.shape[1]
            in_channels = np.zeros(
                (input_height + 2 * zp, input_width + 2 * zp))
            in_channels[zp: zp + input_height,
                        zp: zp + input_width] = input_channels
            return in_channels


# The element_wise_op function implements an element operation on the numpy array and writes the return value back into the array
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


# The Filter class saves the parameters and gradients of the convolutional layer and implements the gradient descent algorithm to update the parameters.
class Filter(object):
    """filter base

      width: img width.
      height: img height.
      depth: img depth.
      filter_num: filter of number.
    """

    def __init__(self, width, height, depth, filter_num):
        wimin = -math.sqrt(6 / (width*height*depth + width*height*filter_num))
        wimax = -wimin
        self.weights = np.random.uniform(
            wimin, wimax, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


# conv2d convolution layer
class Conv2D(object):
    """init conv base function.

      input_width: input img width.
      input_height: input img height.
      channel_number: img channels.
      filter_width: filter width.
      filter_height: filter height.
      filter_number: filter of numbers.
      zero_padding: all zero paddinbg.
      stride: step length.
      activator: Tanh, ReLU, Sigmoid.
      learning_rate: learning rate.
    """

    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, filter_number,
                 zero_padding, stride, activator, learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = int(Conv2D.calculate_output_size(
            self.input_width, filter_width, zero_padding, stride))
        self.output_height = int(Conv2D.calculate_output_size(
            self.input_height, filter_height, zero_padding, stride))
        self.output_channels = np.zeros(
            (self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(
                Filter(filter_width, filter_height, self.channel_number, filter_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_channels):
        self.input_channels = input_channels
        self.padded_input_channels = padding(
            input_channels, self.zero_padding)
        for i in range(self.filter_number):
            filter = self.filters[i]
            conv(self.padded_input_channels, self.output_channels[i], filter.get_weights(),
                 self.stride, filter.get_bias())
        element_wise_op(self.output_channels, self.activator.forward)

    #
    def backward(self, input_channels, sensitivity_array, activator):
        """Calculate the error term passed to the previous layer and calculate the gradient of each weight
           The error terms for the previous layer are stored in self.delta_array
           The gradient is saved in the Filter object's weights_grad
        """

        self.forward(input_channels)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    # update the weights according to the gradient descent
    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)

    # You pass the error term up to the next level.
    # sensitivity_array: layer lost.
    # activator: A layer above the activation function.
    def bp_sensitivity_map(self, sensitivity_array, activator):
        expanded_error_array = self.expand_sensitivity_map(sensitivity_array)
        expanded_width = expanded_error_array.shape[2]
        zp = int((self.input_width + self.filter_width -
                  1 - expanded_width) / 2)
        in_channels = padding(expanded_error_array, zp)
        self.delta_array = self.create_delta_array()
        for i in range(self.filter_number):
            filter = self.filters[i]
            flipped_weights = []
            for oneweight in filter.get_weights():
                flipped_weights.append(np.rot90(oneweight, 2))
            flipped_weights = np.array(flipped_weights)
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(in_channels[i], delta_array[d], flipped_weights[d],  1, 0)
            self.delta_array += delta_array

        derivative_array = np.array(self.input_channels)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        expanded_error_array = self.expand_sensitivity_map(sensitivity_array)
        for i in range(self.filter_number):
            filter = self.filters[i]
            for d in range(filter.weights.shape[0]):
                conv(
                    self.padded_input_channels[d], filter.weights_grad[d], expanded_error_array[i],  1, 0)

            filter.bias_grad = expanded_error_array[i].sum()

    # 0 is added to the positions of sensitivitymap with step size S,
    # which is "restored" to sensitivitymap with step size 1,
    # and then equation 8 is used to solve the problem
    def expand_sensitivity_map(self, sensitivity_array):
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    # Create an array used to save the sensitivity map passed to the previous layer.
    # The output of the previous layer is the input of this layer.
    # So the dimension of the error term at the previous level is the same as the dimension of the input at this level.)
    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1


# max pool 2d base
class MaxPooling2D(object):
    """ maxPool2D base function

      input_width: input img width.
      input_height: input img height.
      channel_number: img channels.
      filter_width: filter width.
      filter_height: filter height.
      stride: step length.
      output_width:  output img width.
      output_height: output img height.
    """

    def __init__(self, input_width, input_height, channel_number, filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / self.stride + 1)
        self.output_height = int(
            (input_height - filter_height) / self.stride + 1)
        self.output_channels = np.zeros(
            (self.channel_number, self.output_height, self.output_width))

    # forward cal.
    def forward(self, input_channels):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_channels[d, i, j] = (get_patch(
                        input_channels[d], i, j, self.filter_width, self.filter_height, self.stride).max())

    # back propagation error.
    def backward(self, input_channels, sensitivity_array):
        self.delta_array = np.zeros(input_channels.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_channels[d], i, j, self.filter_width, self.filter_height, self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + k, j *
                                     self.stride + l] = sensitivity_array[d, i, j]
