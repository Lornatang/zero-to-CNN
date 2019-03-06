import datetime
import numpy as np
import Activators
import CNN
import MNIST
import DNN


class MyNet():
    def __init__(self):
        self.conv1 = CNN.Conv2D(28, 28, 1, 5, 5, 6, 0, 1, Activators.Sigmoid(
        ), 0.02)
        self.maxpool1 = CNN.MaxPooling2D(24, 24, 6, 2, 2, 2)
        self.conv2 = CNN.Conv2D(12, 12, 6, 5, 5, 12, 0, 1, Activators.Sigmoid(
        ), 0.02)
        self.maxpool2 = CNN.MaxPooling2D(8, 8, 12, 2, 2, 2)
        self.fullconnect = DNN.FullConnected(
            192, 10, Activators.Sigmoid(), 0.02)

    def forward(self, onepic):
        self.conv1.forward(onepic)
        self.maxpool1.forward(self.conv1.output_channels)
        self.conv2.forward(self.maxpool1.output_channels)
        self.maxpool2.forward(self.conv2.output_channels)
        flinput = self.maxpool2.output_channels.flatten().reshape(-1, 1)
        self.fullconnect.forward(flinput)
        return self.fullconnect.output

    def backward(self, onepic, labels):
        delta = np.multiply(self.fullconnect.activator.backward(
            self.fullconnect.output), (labels - self.fullconnect.output))

        self.fullconnect.backward(delta)
        self.fullconnect.update()
        sensitivity_array = self.fullconnect.delta.reshape(
            self.maxpool2.output_channels.shape)
        self.maxpool2.backward(self.conv2.output_channels, sensitivity_array)
        self.conv2.backward(self.maxpool1.output_channels,
                            self.maxpool2.delta_array, Activators.Sigmoid())
        self.conv2.update()
        self.maxpool1.backward(
            self.conv1.output_channels, self.conv2.delta_array)
        self.conv1.backward(onepic, self.maxpool1.delta_array,
                            Activators.Sigmoid())
        self.conv1.update()


if __name__ == '__main__':

    train_data_set, train_labels = MNIST.get_training_data_set(
        600, False)
    test_data_set, test_labels = MNIST.get_test_data_set(
        100, False)
    train_data_set = np.array(train_data_set).astype(
        bool).astype(int)
    train_labels = np.array(train_labels)
    test_data_set = np.array(test_data_set).astype(
        bool).astype(int)
    test_labels = np.array(test_labels)
    print("Train numbers:%d" % len(train_data_set))
    print("Test numbers:%d" % len(test_data_set))

    mynetwork = MyNet()

    for i in range(10):
        print('Epoch:', i)
        for k in range(train_data_set.shape[0]):
            onepic = train_data_set[k]
            onepic = np.array([onepic])
            result = mynetwork.forward(onepic)
            labels = train_labels[k].reshape(-1, 1)
            mynetwork.backward(onepic, labels)

    right = 0
    for k in range(test_data_set.shape[0]):

        onepic = test_data_set[k]
        onepic = np.array([onepic])
        labels = test_labels[k].reshape(-1, 1)
        pred_type = result.argmax()
        real_type = labels.argmax()

        if pred_type == real_type:
            right += 1

    print('%s after right ratio is %f' %
          (datetime.datetime.now(), right/test_data_set.shape[0]))
