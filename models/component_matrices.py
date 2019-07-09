import numpy as np
import tensorflow as tf


# Diagonal unitary matrix
class DiagonalMatrix():
    def __init__(self, name, num_units):
        init_w = tf.random_uniform([num_units], minval=-np.pi, maxval=np.pi)
        self.w = tf.Variable(init_w, name=name)
        self.vec = tf.complex(tf.cos(self.w), tf.sin(self.w))

    # [batch_sz, num_units]
    def mul(self, z):
        # [num_units] * [batch_sz, num_units] -> [batch_sz, num_units]
        return self.vec * z


# Reflection unitary matrix
class ReflectionMatrix():
    def __init__(self, name, num_units):
        self.num_units = num_units

        self.re = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name + "_re")
        self.im = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name + "_im")
        self.v = tf.complex(self.re, self.im)  # [num_units]
        self.v = normalize(self.v)
        self.vstar = tf.conj(self.v)  # [num_units]

    # [batch_sz, num_units]
    def mul(self, z):
        v = tf.expand_dims(self.v, 1)  # [num_units, 1]
        vstar = tf.conj(v)  # [num_units, 1]
        vstar_z = tf.matmul(z, vstar)  # [batch_size, 1]
        sq_norm = tf.reduce_sum(tf.abs(self.v) ** 2)  # [1]
        factor = (2 / tf.complex(sq_norm, 0.0))
        return z - factor * tf.matmul(vstar_z, tf.transpose(v))


# Permutation unitary matrix
class PermutationMatrix:
    def __init__(self, name, num_units):
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        self.P = tf.constant(perm, tf.int32)

    # [batch_sz, num_units], permute columns
    def mul(self, z):
        return tf.transpose(tf.gather(tf.transpose(z), self.P))


# FFTs
# z: complex[batch_sz, num_units]

def FFT(z):
    return tf.fft(z, name="fourier")


def IFFT(z):
    return tf.ifft(z, name="inv_fourier")


def normalize(z):
    norm = tf.sqrt(tf.reduce_sum(tf.abs(z) ** 2))
    factor = (norm + 1e-6)
    return tf.complex(tf.real(z) / factor, tf.imag(z) / factor)


# z: complex[batch_sz, num_units]
# bias: real[num_units]
def modReLU(z, bias):  # relu(|z|+b) * (z / |z|)
    norm = tf.abs(z)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    scaled = tf.complex(tf.real(z) * scale, tf.imag(z) * scale)
    return scaled