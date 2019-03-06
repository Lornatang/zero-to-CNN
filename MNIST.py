import numpy as np


# Loader base.
class Loader(object):
    """init
    path: data dir.
    count: file count.
    """

    def __init__(self, path, count):
        self.path = path
        self.count = count

    # read file
    def get_file_content(self):
        print(self.path)
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content  # byte array


# Image data loader
class ImageLoader(Loader):

    def get_picture(self, content, index):
        # file header have 16 bytes.
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                byte1 = content[start + i * 28 + j]
                picture[i].append(byte1)
        return picture

    # convert image data to vector (28 * 28)
    def get_one_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    # loader all data.
    # function: onerow is convert to vector
    def load(self, onerow=False):
        content = self.get_file_content()  # gets an array of file bytes
        data_set = []
        for index in range(self.count):
            onepic = self.get_picture(content, index)
            if onerow:
                onepic = self.get_one_sample(onepic)
            data_set.append(onepic)
        return data_set


# label data loader
class LabelLoader(Loader):
    def load(self):
        content = self.get_file_content()   # gets an array of file bytes
        labels = []
        for index in range(self.count):
            onelabel = content[index + 8]
            onelabelvec = self.norm(onelabel)  # one-hot
            labels.append(onelabelvec)
        return labels

    # internal function, one-hot code. Convert a value to a 10 - dimensional label vector
    def norm(self, label):
        label_vec = []
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec


# get all train data.
def get_training_data_set(num, onerow=False):
    image_loader = ImageLoader(
        'train-images.idx3-ubyte', num)
    label_loader = LabelLoader(
        'train-labels.idx1-ubyte', num)
    return image_loader.load(onerow), label_loader.load()


# get all test data
def get_test_data_set(num, onerow=False):
    image_loader = ImageLoader(
        't10k-images.idx3-ubyte', num)
    label_loader = LabelLoader(
        't10k-labels.idx1-ubyte', num)
    return image_loader.load(onerow), label_loader.load()


# A line vector of 784, printed as a graphic style
def imshow(onepic):
    onepic = onepic.reshape(28, 28)
    for i in range(28):
        for j in range(28):
            if onepic[i, j] == 0:
                print('  ', end='')
            else:
                print('* ', end='')
        print('')


if __name__ == "__main__":
    train_data_set, train_labels = get_training_data_set(
        100)
    train_data_set = np.array(train_data_set)
    train_labels = np.array(train_labels)
    onepic = train_data_set[10]
    imshow(onepic)
    print(train_labels[10].argmax())
