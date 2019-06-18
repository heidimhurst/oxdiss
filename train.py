import tensorflow as tf
from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
from models.tf_rnn import TFRNN
from models.urnn_cell import URNNCell
from models.householder_cell import REFLECTCell
from models.component_matrices import modReLU

import argparse
import numpy as np

'''
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function):
'''

loss_path='results/'

glob_learning_rate = 0.001
glob_decay = 0.9

def baseline_cm(timesteps):
    return 10*np.log(8) / timesteps

def baseline_ap():
    return 0.167

def serialize_loss(loss, name):
    file=open(loss_path + name, 'w')
    for l in loss:
        file.write("{0}\n".format(l))

class Main:
    def init_data(self, adding_problem, memory_problem, batch_size=50, epochs=10, seed=True):
        tf.logging.info('Generating data...')

        if memory_problem:
            tf.logging.info('Generating memory problem data...')
            # init copying memory problem
            self.cm_batch_size=batch_size
            self.cm_epochs=epochs

            self.cm_timesteps=[120, 220, 320, 520]
            self.cm_samples=100000
            self.cm_data=[CopyingMemoryProblemDataset(self.cm_samples, timesteps, seed) for timesteps in self.cm_timesteps]
            self.dummy_cm_data=CopyingMemoryProblemDataset(100, 50) # samples, timestamps

        if adding_problem:
            tf.logging.info('Generating adding problem data...')
            # init adding problem
            self.ap_batch_size=batch_size
            self.ap_epochs=epochs

            # self.ap_timesteps=[100, 200, 400, 750]
            # self.ap_samples=[30000, 50000, 40000, 100000]

            # small for testing
            self.ap_timesteps = [50]
            self.ap_samples = [200]

            self.ap_data=[AddingProblemDataset(sample, timesteps, seed) for
                          timesteps, sample in zip(self.ap_timesteps, self.ap_samples)]
            self.dummy_ap_data=AddingProblemDataset(100, 50) # samples, timestamps

        tf.logging.info('Done.')

    def train_network(self, net, dataset, batch_size, epochs):

        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)

            sample_len = str(dataset.get_sample_len())
            tf.logging.info('Training network {} ... timesteps= {}'.format(net.name,sample_len))
            # net.summary()
            net.train(dataset, batch_size, epochs)
            # loss_list has one number for each batch (step)
            serialize_loss(net.get_loss_list(), net.name + sample_len)
            # todo: get accuracy, not just loss

            save_path = saver.save(sess, "./tmp/model_{}.ckpt".format(net.name))
            print("Model saved in path: %s" % save_path)

            # todo: move testing to somewhere else in the code
            tf.logging.info("TESTING just for shits to see if it works")
            net.test(dataset)

        tf.logging.info('Training network {} done.'.format(net.name))

    def train_urnn_for_timestep_idx(self, idx, adding_problem, memory_problem):
        tf.logging.info('Initializing and training URNNs for one timestep...')

        if memory_problem:
            tf.logging.info('Training urnn for memory problem...')
            # CM
            tf.reset_default_graph()
            self.cm_urnn=TFRNN(
                name="cm_urnn",
                num_in=1,
                num_hidden=128,
                num_out=10,
                num_target=1,
                single_output=False,
                rnn_cell=URNNCell,
                activation_hidden=None, # modReLU
                # activation_hidden= modReLU, # this doesn't change anything as the modReLU is included in the cell by default
                activation_out=tf.identity,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
            self.train_network(self.cm_urnn, self.cm_data[idx],
                               self.cm_batch_size, self.cm_epochs)

        if adding_problem:
            tf.logging.info('Training urnn for adding problem ...')
            # AP
            tf.reset_default_graph()
            self.ap_urnn=TFRNN(
                name="ap_urnn",
                num_in=2,
                num_hidden=512,
                num_out=1,
                num_target=1,
                single_output=True,
                rnn_cell=URNNCell,
                activation_hidden=None, # modReLU
                # activation_hidden= modReLU, # this doesn't change anything as the modReLU is included in the cell by default
                activation_out=tf.identity,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                loss_function=tf.squared_difference)
            self.train_network(self.ap_urnn, self.ap_data[idx],
                               self.ap_batch_size, self.ap_epochs)

        tf.logging.info('Init and training URNNs for one timestep done.')


    def train_rnn_lstm_for_timestep_idx(self, idx, adding_problem, memory_problem):
        tf.logging.info('Initializing and training RNN&LSTM for one timestep...')

        if memory_problem:
            tf.logging.info("Training rnn/lstm for memory problem ...")
            # CM

            tf.reset_default_graph()
            self.cm_simple_rnn=TFRNN(
                name="cm_simple_rnn",
                num_in=1,
                num_hidden=80,
                num_out=10,
                num_target=1,
                single_output=False,
                rnn_cell=tf.contrib.rnn.BasicRNNCell,
                activation_hidden=tf.tanh,
                activation_out=tf.identity,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
            self.train_network(self.cm_simple_rnn, self.cm_data[idx],
                               self.cm_batch_size, self.cm_epochs)

            tf.reset_default_graph()
            self.cm_lstm=TFRNN(
                name="cm_lstm",
                num_in=1,
                num_hidden=40,
                num_out=10,
                num_target=1,
                single_output=False,
                rnn_cell=tf.contrib.rnn.LSTMCell,
                activation_hidden=tf.tanh,
                activation_out=tf.identity,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
            self.train_network(self.cm_lstm, self.cm_data[idx],
                               self.cm_batch_size, self.cm_epochs)

        if adding_problem:
            tf.logging.info("Training rnn/lstm for adding problem ...")
            # AP
            tf.reset_default_graph()
            self.ap_simple_rnn=TFRNN(
                name="ap_simple_rnn",
                num_in=2,
                num_hidden=128,
                num_out=1,
                num_target=1,
                single_output=True,
                rnn_cell=tf.contrib.rnn.BasicRNNCell,
                activation_hidden=tf.tanh,
                activation_out=tf.identity,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                loss_function=tf.squared_difference)
            self.train_network(self.ap_simple_rnn, self.ap_data[idx],
                               self.ap_batch_size, self.ap_epochs)

            tf.reset_default_graph()
            self.ap_lstm=TFRNN(
                name="ap_lstm",
                num_in=2,
                num_hidden=128,
                num_out=1,
                num_target=1,
                single_output=True,
                rnn_cell=tf.contrib.rnn.LSTMCell,
                activation_hidden=tf.tanh,
                activation_out=tf.identity,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                loss_function=tf.squared_difference)
            self.train_network(self.ap_lstm, self.ap_data[idx],
                               self.ap_batch_size, self.ap_epochs)

        tf.logging.info('Init and training networks for one timestep done.')

    def train_networks(self, adding_problem, memory_problem, urnn=True, lstm=False, timesteps_idx=1):
        tf.logging.info('Starting training...')

        # timesteps_idx=4
        if urnn:
            for i in range(timesteps_idx):
                main.train_urnn_for_timestep_idx(i, adding_problem, memory_problem)
        if lstm:
            for i in range(timesteps_idx):
                main.train_rnn_lstm_for_timestep_idx(i, adding_problem, memory_problem)

        tf.logging.info('Done and done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-tr", '--train', dest='train', action='store_true')

    parser.add_argument("-a", '--adding-problem', dest="adding_problem", action='store_true')
    parser.add_argument("-m", '--memory-problem', dest="memory_problem", action='store_true')

    parser.add_argument("-b", '--batch-size', dest="batch_size", type=int, default=50)
    parser.add_argument("-e", '--epochs', dest="epochs", type=int, default=10)

    # train urnn
    parser.add_argument("-u", '--urnn', dest="urnn", action="store_true")
    # train lstm & rnn
    parser.add_argument("-l", "--lstm", dest="lstm", action="store_true")
    # generate random data (i.e. not from seed)
    parser.add_argument("-r", '--randomize-data', dest="seed", action="store_false")

    args = parser.parse_args()

    # set logging verbosity to view commands/info
    tf.logging.set_verbosity(tf.logging.INFO)

    main=Main()
    main.init_data(args.adding_problem, args.memory_problem, args.batch_size, args.epochs, args.seed)

    if args.train:
        main.train_networks(args.adding_problem, args.memory_problem, args.urnn, args.lstm)

    # main.test_networks(args.adding_problem, args.memory_problem)