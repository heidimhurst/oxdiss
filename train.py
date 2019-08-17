import tensorflow as tf

from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
from problems.mnist import MnistProblemDataset
from models.tf_rnn import TFRNN
from models.urnn_cell import URNNCell
from models.householder_cell import REFLECTCell
from models.rd_cell import RDCell
from models.prd_cell import PRDCell
from models.frpdi_cell import FRPDICell

from models.component_matrices import modReLU

from glob import glob
from datetime import datetime

import socket

import argparse
import numpy as np
import os

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

loss_path = 'results/'

glob_learning_rate = 0.001
glob_decay = 0.9

default_options = {"adding_problem": True,
                   "memory_problem": True,
                   "mnist_problem": False,
                   "permuted_mnist": False,
                   "batch_size": 128,
                   "epochs": 2,
                   "seed": False,
                   "urnn": True,
                   "lstm": False,
                   "rnn": False,
                   "log_output": "default",
                   # "output":args.output,
                   "cell_type": "urnn",
                   "optimization": "adam",
                   "checkpoints": 0,
                   "resume":""}

# specify optimization scheme
optimizers = {"rmsprop": tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
              "adam": tf.train.AdamOptimizer(learning_rate=glob_learning_rate),
              "adadelta": tf.train.AdadeltaOptimizer(learning_rate=glob_learning_rate),
              "kfac":"kfac",
              "adaqn":"adaqn"}


def baseline_cm(timesteps):
    return 10*np.log(8) / timesteps


def baseline_ap():
    return 0.167


def serialize_loss(loss, path, name):
    with open(os.path.join(path, name+".txt"), "w") as file:
        for l in loss:
            file.write("{0}\n".format(l))


class Main:

    def init_data(self, options=default_options):

        self.output_info = {"run_date":datetime.now().strftime("%x"),
                            "start_time": datetime.now().strftime("%X"),
                            "learning_rate": glob_learning_rate,
                            "decay": glob_decay,
                            "machine": socket.gethostname(),
                            "checkpoints": options["checkpoints"]}

        tf.logging.info('Generating data...')

        if options["memory_problem"]:
            tf.logging.info('Generating memory problem data...')
            # init copying memory problem
            self.cm_batch_size=options["batch_size"]
            self.cm_epochs=options["epochs"]

            self.cm_timesteps=[120, 220, 320, 520]
            self.cm_samples=100000
            self.cm_data=[CopyingMemoryProblemDataset(self.cm_samples, timesteps, options["seed"]) for timesteps in self.cm_timesteps]
            self.dummy_cm_data=CopyingMemoryProblemDataset(100, 50) # samples, timestamps

        if options["adding_problem"]:
            tf.logging.info('Generating adding problem data...')
            # init adding problem
            self.ap_batch_size=options["batch_size"]
            self.ap_epochs=options["epochs"]

            # self.ap_timesteps=[100, 200, 400, 750]
            # self.ap_samples=[30000, 50000, 40000, 100000]

            # small for testing
            self.ap_timesteps = [100]
            self.ap_samples = [50000]

            self.ap_data=[AddingProblemDataset(sample, timesteps, options["seed"]) for
                          timesteps, sample in zip(self.ap_timesteps, self.ap_samples)]
            self.dummy_ap_data=AddingProblemDataset(100, 50) # samples, timestamps

        if options["mnist_problem"]:
            tf.logging.info('Reading in MNIST problem data...')

            self.mnist_batch_size = options["batch_size"]
            self.mnist_epochs = options["epochs"]
            self.mnist_data = MnistProblemDataset(-1, -1, options["permuted_mnist"])

            # write info to dictionary for later use
            self.output_info["batch_size"] = self.mnist_batch_size
            self.output_info["epochs"] = self.mnist_epochs


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
            serialize_loss(net.get_loss_list(), net.log_dir, net.name + sample_len)
            # todo: get accuracy, not just loss

            save_path = saver.save(sess, os.path.join(net.log_dir,"model_{}.ckpt".format(net.name)))
            print("Final model checkpoint saved in path: %s" % save_path)

            # todo: move testing to somewhere else in the code
            tf.logging.info("TESTING just for shits to see if it works")
            net.test(dataset)
            # TODO: write this result out to the info.txt file
            

        tf.logging.info('Training network {} done.'.format(net.name))

    def train_urnn_for_timestep_idx(self, idx,
                                    options=default_options):

        # specify cell information
        cells = {"urnn": URNNCell, "householder": REFLECTCell,
                 "rd": RDCell, "prd": PRDCell, "frpdi": FRPDICell}

        if options["cell_type"] in cells.keys():
            cell = cells[options["cell_type"]]
        else:
            print("GOING FOR STRING")
            cell = options["cell_type"]

        tf.logging.info('Initializing and training URNNs for one timestep...')



        if options["memory_problem"]:
            # write info to dictionary for later use
            self.output_info["batch_size"] = self.cm_batch_size
            self.output_info["epochs"] = self.cm_epochs
            self.output_info["timesteps"] = self.cm_timesteps[idx]
            self.output_info["samples"] = self.cm_samples

            tf.logging.info('Training urnn for memory problem...')
            # CM
            tf.reset_default_graph()
            self.cm_urnn=TFRNN(
                name="cm_{}".format(options["cell_type"]),
                log_output=options["log_output"],
                num_in=1,
                num_hidden=128,
                num_out=10,
                num_target=1,
                single_output=False,
                rnn_cell=cell,
                activation_hidden=None,  # modReLU
                # activation_hidden= modReLU, # this doesn't change anything as the modReLU is included in the cell by default
                activation_out=tf.identity,
                optimizer=optimizers[options["optimization"]],
                loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                output_info=self.output_info,
                checkpoints=options["checkpoints"],
                resume=options["resume"])
            self.train_network(self.cm_urnn, self.cm_data[idx],
                               self.cm_batch_size, self.cm_epochs)

        if options["adding_problem"]:
            # write info to dictionary for later use
            self.output_info["batch_size"] = self.ap_batch_size
            self.output_info["epochs"] = self.ap_epochs
            self.output_info["timesteps"] = self.ap_timesteps[idx]
            self.output_info["samples"] = self.ap_samples[0]

            tf.logging.info('Training urnn for adding problem ...')
            # AP
            tf.reset_default_graph()
            self.ap_urnn=TFRNN(
                name="ap_{}".format(options["cell_type"]),
                log_output=options["log_output"],
                num_in=2,
                num_hidden=512,
                num_out=1,
                num_target=1,
                single_output=True,
                rnn_cell=cell,
                activation_hidden=None, # modReLU
                # activation_hidden= modReLU, # this doesn't change anything as the modReLU is included in the cell by default
                activation_out=tf.identity,
                # optimizer=tf.train.RMSPropOptimizer(learning_rate=glob_learning_rate, decay=glob_decay),
                optimizer=optimizers[options["optimization"]],
                loss_function=tf.squared_difference,
                output_info=self.output_info,
                checkpoints=options["checkpoints"],
                resume=options["resume"])
            self.train_network(self.ap_urnn, self.ap_data[idx],
                               self.ap_batch_size, self.ap_epochs)

        if options["mnist_problem"]:
            # # write info to dictionary for later use
            # self.output_info["batch_size"] = self.mnist_batch_size
            # self.output_info["epochs"] = self.mnist_epochs


            tf.reset_default_graph()
            self.mnist_urnn = TFRNN(
                name="mnist_{}".format(options["cell_type"]),
                log_output=options["log_output"],
                num_in=1,
                num_hidden=512,
                num_out=10,
                num_target=1,
                single_output=True,
                rnn_cell=cell,
                activation_hidden=None,  # modReLU
                activation_out=tf.identity,
                optimizer=optimizers[options["optimization"]],
                loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                output_info=self.output_info,
                checkpoints=options["checkpoints"],
                resume=options["resume"])
            self.train_network(self.mnist_urnn, self.mnist_data,
                               self.mnist_batch_size, self.mnist_epochs)

        tf.logging.info('Init and training URNNs for one timestep done.')

    def train_rnn_lstm_for_timestep_idx(self, idx, options=default_options):

        self.output_info = {"run_date":datetime.now().strftime("%x"),
                            "start_time": datetime.now().strftime("%X"),
                            "learning_rate": glob_learning_rate,
                            "decay": glob_decay}

        tf.logging.info('Initializing and training RNN&LSTM for one timestep...')

        if options["memory_problem"]:
            # write info to dictionary for later use
            self.output_info["batch_size"] = self.cm_batch_size
            self.output_info["epochs"] = self.cm_epochs
            self.output_info["timesteps"] = self.cm_timesteps[idx]
            self.output_info["samples"] = self.cm_samples

            tf.logging.info("Training rnn/lstm for memory problem ...")
            # CM

            if options["rnn"]:
                tf.reset_default_graph()
                self.cm_simple_rnn=TFRNN(
                    name="cm_simplernn",
                    log_output=options["log_output"],
                    num_in=1,
                    num_hidden=80,
                    num_out=10,
                    num_target=1,
                    single_output=False,
                    rnn_cell=tf.contrib.rnn.BasicRNNCell,
                    activation_hidden=tf.tanh,
                    activation_out=tf.identity,
                    optimizer=optimizers[options["optimization"]],
                    loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                    output_info=self.output_info,
                    checkpoints=options["checkpoints"],
                    resume=options["resume"])
                self.train_network(self.cm_simple_rnn, self.cm_data[idx],
                                   self.cm_batch_size, self.cm_epochs)

            if options["lstm"]:
                tf.reset_default_graph()
                self.cm_lstm=TFRNN(
                    name="cm_lstm",
                    log_output=options["log_output"],
                    num_in=1,
                    num_hidden=40,
                    num_out=10,
                    num_target=1,
                    single_output=False,
                    rnn_cell=tf.contrib.rnn.LSTMCell,
                    activation_hidden=tf.tanh,
                    activation_out=tf.identity,
                    optimizer=optimizers[options["optimization"]],
                    loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                    output_info=self.output_info,
                    checkpoints=options["checkpoints"],
                    resume=options["resume"])
                self.train_network(self.cm_lstm, self.cm_data[idx],
                                   self.cm_batch_size, self.cm_epochs)

        if options["adding_problem"]:
            # write info to dictionary for later use
            self.output_info["batch_size"] = self.ap_batch_size
            self.output_info["epochs"] = self.ap_epochs
            self.output_info["timesteps"] = self.ap_timesteps[idx]
            self.output_info["samples"] = self.ap_samples[0]

            tf.logging.info("Training rnn/lstm for adding problem ...")
            # AP
            if options["rnn"]:
                tf.reset_default_graph()
                self.ap_simple_rnn=TFRNN(
                    name="ap_simplernn",
                    log_output=options["log_output"],
                    num_in=2,
                    num_hidden=128,
                    num_out=1,
                    num_target=1,
                    single_output=True,
                    rnn_cell=tf.contrib.rnn.BasicRNNCell,
                    activation_hidden=tf.tanh,
                    activation_out=tf.identity,
                    optimizer=optimizers[options["optimization"]],
                    loss_function=tf.squared_difference,
                    output_info=self.output_info,
                    checkpoints=options["checkpoints"],
                    resume=options["resume"])
                self.train_network(self.ap_simple_rnn, self.ap_data[idx],
                                   self.ap_batch_size, self.ap_epochs)

            if options["lstm"]:
                tf.reset_default_graph()
                self.ap_lstm=TFRNN(
                    name="ap_lstm",
                    log_output=options["log_output"],
                    num_in=2,
                    num_hidden=128,
                    num_out=1,
                    num_target=1,
                    single_output=True,
                    rnn_cell=tf.contrib.rnn.LSTMCell,
                    activation_hidden=tf.tanh,
                    activation_out=tf.identity,
                    optimizer=optimizers[options["optimization"]],
                    loss_function=tf.squared_difference,
                    output_info=self.output_info,
                    checkpoints=options["checkpoints"],
                    resume=options["resume"])
                self.train_network(self.ap_lstm, self.ap_data[idx],
                                   self.ap_batch_size, self.ap_epochs)

        if options["mnist_problem"]:

            # write info to dictionary for later use
            self.output_info["batch_size"] = self.mnist_batch_size
            self.output_info["epochs"] = self.mnist_epochs


            if options["rnn"]:
                tf.reset_default_graph()
                self.mnist_lstm=TFRNN(
                    name="mnist_simplernn",
                    log_output=options["log_output"],
                    num_in=1,
                    num_hidden=128,
                    num_out=10,
                    num_target=1,
                    single_output=True,
                    rnn_cell=tf.contrib.rnn.BasicRNNCell,
                    activation_hidden=tf.tanh,
                    activation_out=tf.identity,
                    optimizer=optimizers[options["optimization"]],
                    loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                    output_info=self.output_info,
                    checkpoints=options["checkpoints"],
                    resume=options["resume"])
                self.train_network(self.mnist_lstm, self.mnist_data,
                                   self.mnist_batch_size, self.mnist_epochs)

            if options["lstm"]:
                tf.reset_default_graph()
                self.mnist_lstm=TFRNN(
                    name="mnist_lstm",
                    log_output=options["log_output"],
                    num_in=1,
                    num_hidden=128,
                    num_out=10,
                    num_target=1,
                    single_output=True,
                    rnn_cell=tf.contrib.rnn.LSTMCell,
                    activation_hidden=tf.tanh,
                    activation_out=tf.identity,
                    optimizer=optimizers[options["optimization"]],
                    loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
                    output_info=self.output_info,
                    checkpoints=options["checkpoints"],
                    resume=options["resume"])
                self.train_network(self.mnist_lstm, self.mnist_data,
                                   self.mnist_batch_size, self.mnist_epochs)

        tf.logging.info('Init and training networks for one timestep done.')

    def train_networks(self, timesteps_idx=1, options=default_options):
        tf.logging.info('Starting training...')

        print(options)


        # timesteps_idx=4
        if options["urnn"]:
            for i in range(timesteps_idx):
                main.train_urnn_for_timestep_idx(i, options=options)
        if options["lstm"] or options["rnn"]:
            for i in range(timesteps_idx):
                main.train_rnn_lstm_for_timestep_idx(i, options=options)

        tf.logging.info('Done and done.')


def increment_trial(logs="./logs/"):
    """

    :param logs: Folder containing log directories (one directory per trial ideally)
    :return:
    """
    # get log data
    folders = glob(logs+"*/")

    # cut to just folder name
    folders = [fold[len(logs):-1] for fold in folders]

    # cut to exclude everything after first underscore
    folders = [fold.split("_")[0] for fold in folders]

    # get largest integer
    next_trial = max([int(fold) for fold in folders if fold.isdigit()]) + 1

    return str(next_trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-tr", '--train', dest='train', action='store_true')

    parser.add_argument("-a", '--adding-problem', dest="adding_problem", action='store_true')
    parser.add_argument("-m", '--memory-problem', dest="memory_problem", action='store_true')
    parser.add_argument("-i", '--mnist-problem', dest="mnist_problem", action='store_true')
    parser.add_argument("-p", '--permuted-mnist', dest="permuted_mnist", action='store_true')

    parser.add_argument("-b", '--batch-size', dest="batch_size", type=int, default=128)
    parser.add_argument("-e", '--epochs', dest="epochs", type=int, default=2)

    # save frequency checkpoints (0 will save only at the end, otherwise will save every n steps)
    parser.add_argument("-s", '--checkpoints', dest="checkpoints", type=int, default=0)

    # specify cell type for URNN (options at present are 'householder', 'urnn', 'rd')
    parser.add_argument("-c", '--cell-type', dest="cell_type", type=str, default="urnn")
    # specify optimization scheme (options at present are 'adam' 'adagrad' 'rmsprop')
    parser.add_argument("-o", '--optimization-method', dest="optimization", type=str, default='adam')

    # train urnn
    parser.add_argument("-u", '--urnn', dest="urnn", action="store_true")
    # train lstm & rnn
    parser.add_argument("-l", "--lstm", dest="lstm", action="store_true")
    parser.add_argument("-n", "--rnn", dest="rnn", action="store_true")
    # generate random data (i.e. not from seed)
    # parser.add_argument("-r", '--randomize-data', dest="seed", action="store_true")

    # storage location for runs
    # parser.add_argument("-o", "--output", dest="output", type=str)

    # restart from checkpoint
    parser.add_argument("-r", "--resume", dest="resume", type=str, default="")

    args = parser.parse_args()

    # enable eager execution for ease of printing/debugging
    # tf.compat.v1.enable_eager_execution()

    # set logging verbosity to view commands/info
    tf.logging.set_verbosity(tf.logging.INFO)

    # print cell type info
    # tf.logging.info("URNN cell type set to {}".format(args.cell_type))

    # if permuted mnist, require mnist
    if args.permuted_mnist:
        args.mnist_problem = True

    input_options = {"adding_problem":args.adding_problem,
                        "memory_problem":args.memory_problem,
                        "mnist_problem":args.mnist_problem,
                        "permuted_mnist":args.permuted_mnist,
                        "batch_size":args.batch_size,
                        "epochs":args.epochs,
                        # "seed":args.seed,
                        "urnn":args.urnn,
                        "lstm":args.lstm,
                        "rnn":args.rnn,
                        "log_output":"default",
                        # "output":args.output,
                        "cell_type":args.cell_type,
                        "optimization":args.optimization,
                        "checkpoints":args.checkpoints,
                        "resume":args.resume}


    input_options = dict(default_options, **input_options)

    print(input_options)

    main = Main()
    main.init_data(options=input_options)

    if args.train:
        main.train_networks(options=input_options)

    # main.test_networks(args.adding_problem, args.memory_problem)
