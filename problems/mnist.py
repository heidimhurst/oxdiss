import numpy as np
from .dataset import Dataset
from tensorflow.examples.tutorials.mnist import input_data

# single output = False
# num_in = 784
# num_target = 1

class MnistProblemDataset(Dataset):

    def __init__(self, num_samples, sample_len, permute=False, seed=420):
        """

        :param num_samples:
        :param sample_len:
        :param permute: If True, randomly permutes elements of MNIST data set using one permutation matrix
        """
        super(MnistProblemDataset,self).__init__(num_samples, sample_len)
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

        # code for permuted pixelwise MNIST problem
        if permute:
            # create permutation matrix
            permutation_matrix = np.eye(784)
            # ensure that seed is set for transferability across different runs
            if seed is not None:
                np.random.seed(seed)
            # shuffle rows of identity matrix to create permutation matrix
            np.random.shuffle(permutation_matrix)
            # permute training images
            mnist_train_images = self.permute_collection(mnist.train.images, permutation_matrix)
            self.X_train = np.reshape(mnist_train_images, [-1, 784, 1])
            # permute validation images
            mnist_validation_images = self.permute_collection(mnist.validation.images, permutation_matrix)
            self.X_valid = np.reshape(mnist_validation_images, [-1, 784, 1])
            # permute test images
            mnist_test_images = self.permute_collection(mnist.test.images, permutation_matrix)
            self.X_test = np.reshape(mnist_test_images, [-1, 784, 1])
        else:
            self.X_train = np.reshape(mnist.train.images, [-1, 784, 1])
            self.X_valid = np.reshape(mnist.validation.images, [-1, 784, 1])
            self.X_test = np.reshape(mnist.test.images, [-1, 784, 1])

        self.Y_train = np.reshape(mnist.train.labels, [-1, 1])
        self.Y_valid = mnist.validation.labels
        self.Y_test = mnist.test.labels

    def generate(self, num_samples):
        pass

    def permute_collection(self, collection, permutation_matrix):
        for i in range(collection.shape[0]):
            collection[i] = permutation_matrix.dot(collection[i])

        return collection
