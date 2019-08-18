from adversarial.logits import CheckpointInference

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data  # (not in TF2)

import tensorflow as tf

# import tensorflow_datasets as tfds #tf2

from adversarial.fast_gradient_method import fast_gradient_method as fgm


if __name__ == "__main__":

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    # mnist = tfds.load(name="mnist", split=tfds.Split.TEST) # tf2

    folder = "logs/mnist/u/08_15/20/logs" # "test_logs/"
    ckpt_numb = 20  # 8500

    # test checkpoint building
    ckpt = CheckpointInference(folder, ckpt_numb)

    # use MNIST images (correctly reshaped) as input tensors
    input_images = np.reshape(mnist.test.images, [-1, 784, 1])

    # select subset to work with (single reshaped image)
    max_batch_size = 1
    np.random.seed(420240)
    inds = np.random.randint(0, len(input_images), max_batch_size)
    input_subset = tf.convert_to_tensor(input_images[inds, :, :])

    # test logit getting
    logits = ckpt.get_logits(input_subset)

    # test prediction getting
    predictions = ckpt.get_predictions_from_logits(logits)
    predictions2 = ckpt.get_predictions(input_subset)
    print(predictions == predictions2)

    # test Fast Gradient Method
    eps = 0.25
    norm = np.inf

    adversarial_output = fgm(ckpt.get_probabilities, input_subset, eps, norm)







