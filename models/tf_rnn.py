import numpy as np
import tensorflow as tf
from .urnn_cell import URNNCell
from .householder_cell import REFLECTCell
from .rd_cell import RDCell
from .prd_cell import PRDCell
from .frpdi_cell import FRPDICell
from .flexi_cell import FLEXICell

from datetime import datetime
from glob import glob
import os
import json

import sys


def serialize_to_file(loss, outfolder=""):
    file=open(os.path.join(outfolder,'losses.txt'), 'w')
    for l in loss:
        file.write("{0}\n".format(l))
    file.close()



def increment_trial(logdir):
    # creates a new folder for output logs based on time & date
    # format: MM_DD_RR where rr is the run index

    # create folder prefix with date
    folder_prefix = "{:02d}_{:02d}".format(datetime.now().month, datetime.now().day)

    # find matching folders
    folders = glob(os.path.join(logdir, "{}/*".format(folder_prefix)))

    # get just trial numbers
    folders = [fold.split("/")[-1] for fold in folders]
    trial_numbs = [int(fold) for fold in folders if fold.isdigit()]
    trial_numbs.append(0)
    next_trial = max(trial_numbs) + 1

    # create trial name
    trial_name = "{}/{}".format(folder_prefix, next_trial)

    return trial_name




class TFRNN:
    def __init__(
        self,
        name,
        log_output,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer,
        loss_function,
        output_info={}):

        # self
        self.name = name
        self.problem = self.name.split("_")[0]
        self.net_type = self.name.split("_")[1]
        self.loss_list = []
        self.init_state_C = np.sqrt(3 / (2 * num_hidden))
        # self.log_dir = './logs/{}_{}/logs'.format(self.log_output, self.name)
        top_dir = "./logs/{}/{}/".format(self.problem, self.net_type)
        self.log_dir = "./logs/{}/{}/{}/logs".format(self.problem, self.net_type, increment_trial(top_dir))
        self.log_output = self.log_dir[:-5]
        self.writer = tf.summary.FileWriter(self.log_dir)

        # init cell
        if isinstance(rnn_cell, str):
            self.cell = FLEXICell(num_units = num_hidden, num_in = num_in, components=rnn_cell)
        elif rnn_cell == URNNCell or REFLECTCell or RDCell or PRDCell or FRPDICell:
            self.cell = rnn_cell(num_units = num_hidden, num_in = num_in)
        else:
            self.cell = rnn_cell(num_units = num_hidden, activation = activation_hidden)

        # extract output size
        self.output_size = self.cell.output_size
        if type(self.output_size) == dict:
            self.output_size = self.output_size['num_units']
      
        # input_x: [batch_size, max_time, num_in]
        # input_y: [batch_size, max_time, num_target] or [batch_size, num_target]
        self.input_x = tf.placeholder(tf.float32, [None, None, num_in], name="input_x_"+self.name)
        self.input_y = tf.placeholder(tf.float32, [None, num_target] if single_output else [None, None, num_target],  
                                      name="input_y_"+self.name)
        
        # rnn initial state(s)
        self.init_states = []
        self.dyn_rnn_init_states = None

        # prepare state size list
        if type(self.cell.state_size) == int:
            state_size_list = [self.cell.state_size]
            self.dyn_rnn_init_states = self.cell.state_size # prepare init state for dyn_rnn
        elif type(self.cell.state_size) == tf.contrib.rnn.LSTMStateTuple:
            state_size_list = list(self.cell.state_size)
            self.dyn_rnn_init_states = self.cell.state_size # prepare init state for dyn_rnn

        # construct placeholder list==
        for state_size in state_size_list:
            init_state = tf.placeholder(tf.float32, [None, state_size], name="init_state")
            self.init_states.append(init_state)
        
        # prepare init state for dyn_rnn
        if type(self.cell.state_size) == int:
            self.dyn_rnn_init_states = self.init_states[0]
        elif type(self.cell.state_size) == tf.contrib.rnn.LSTMStateTuple:
            self.dyn_rnn_init_states = tf.contrib.rnn.LSTMStateTuple(self.init_states[0], self.init_states[1])

        with tf.name_scope("hidden_to_output"):
            # set up h->o parameters
            self.w_ho = tf.get_variable("w_ho_"+self.name, shape=[num_out, self.output_size],
                                                initializer=tf.contrib.layers.xavier_initializer())  # fixme
            tf.summary.histogram("w_ho_"+self.name, self.w_ho)  # tensorboard
            self.b_o = tf.Variable(tf.zeros([num_out, 1]), name="b_o_"+self.name)
            tf.summary.histogram("b_o_" + self.name, self.b_o)  # tensorboard

        # run the dynamic rnn and get hidden layer outputs
        # outputs_h: [batch_size, max_time, self.output_size]
        outputs_h, final_state = tf.nn.dynamic_rnn(self.cell, self.input_x, initial_state=self.dyn_rnn_init_states) 
        # returns (outputs, state)
        #print("after dyn_rnn outputs_h:", outputs_h.shape, outputs_h.dtype)
        #print("after dyn_rnn final_state:", final_state.shape, final_state.dtype)

        # produce final outputs from hidden layer outputs
        if single_output:
            outputs_h = tf.reshape(outputs_h[:, -1, :], [-1, self.output_size])
            # outputs_h: [batch_size, self.output_size]
            preact = tf.matmul(outputs_h, tf.transpose(self.w_ho)) + tf.transpose(self.b_o)
            outputs_o = activation_out(preact) # [batch_size, num_out]
        else:
            # outputs_h: [batch_size, max_time, m_out]
            out_h_mul = tf.einsum('ijk,kl->ijl', outputs_h, tf.transpose(self.w_ho))
            preact = out_h_mul + tf.transpose(self.b_o)
            outputs_o = activation_out(preact) # [batch_size, time_step, num_out]

        print("outputs_o")
        tf.print(outputs_o, output_stream=tf.compat.v1.logging.warning)
        # tf.Print(outputs_o, [outputs_o])
        print("==endoutputs==")
            
        # calculate losses and set up optimizer

        # loss function is usually one of these two:
        #   tf.nn.sparse_softmax_cross_entropy_with_logits 
        #     (classification, num_out = num_classes, num_target = 1)
        #   tf.squared_difference 
        #     (regression, num_out = num_target)
        if loss_function == tf.squared_difference:
            self.total_loss = tf.reduce_mean(loss_function(outputs_o, self.input_y))
        elif loss_function == tf.nn.sparse_softmax_cross_entropy_with_logits:
            prepared_labels = tf.cast(tf.squeeze(self.input_y), tf.int32)
            self.total_loss = tf.reduce_mean(loss_function(logits=outputs_o, labels=prepared_labels))
        else:
            raise Exception('New loss function')
        self.train_step = optimizer.minimize(self.total_loss, name='Optimizer')

        # tensorboard
        self.writer.add_graph(tf.get_default_graph())
        self.writer.flush()
        self.writer.close()

        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        all_params = tf.trainable_variables()
        tf.print(self.w_ho)
        print(self.w_ho)
        # print(self.w_ho.eval())

        print('Network __init__ over. Number of trainable params=', t_params)

        # output info for later use
        self.start_time = datetime.now()
        self.output_info = output_info
        self.output_info.update({"name": self.name, "net_type": self.net_type, "problem":self.problem,
                                 "directory": self.log_dir, "optimizer": optimizer.get_name(),
                                 "num_in": num_in, "num_out": num_out, "num_hidden": num_hidden,
                                 "num_target": num_target, "cell_type": self.cell._base_name,
                                 "trainable_params": int(t_params)})

        # save info about trial run for later use/analysis/recordkeeping
        self.save_output_info()

    def train(self, dataset, batch_size, epochs):

        # session
        with tf.Session() as sess:
            # print(self.w_ho.eval())
            # initialize loss
            # batch_loss = tf.Variable(0., name="batch_loss")
            tf.summary.scalar("total_loss", self.total_loss)
            # validation_loss = 0
            # tf.summary.scalar("validation_loss", validation_loss)

            counter = 0;
            train_writer = tf.summary.FileWriter('{}/train'.format(self.log_output), sess.graph)
            # train_writer = tf.contrib.summary('./logs/{}_{}/train'.format(self.log_output, self.name), sess.graph)

            # initialize global vars
            sess.run(tf.global_variables_initializer())

            # fetch validation and test sets
            num_batches = dataset.get_batch_count(batch_size)
            X_val, Y_val = dataset.get_validation_data()

            # init loss list
            self.loss_list = []
            tf.logging.info("Starting training for {}".format(self.name))
            tf.logging.info("NumEpochs: {0:3d} |  BatchSize: {1:3d} |  NumBatches: {2:5d} \n".format(epochs,
                                                                                              batch_size,
                                                                                              num_batches))

            # train for several epochs
            for epoch_idx in range(epochs):
                tf.logging.info("Epoch Starting:{} \n ".format(epoch_idx))
                # train on several minibatches
                for batch_idx in range(num_batches):
                    counter += 1 # for use with summary writer
                    merge = tf.summary.merge_all()

                    # get one batch of data
                    # X_batch: [batch_size x time x num_in]
                    # Y_batch: [batch_size x time x num_target] or [batch_size x num_target] (single_output?)
                    X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)



                    # evaluate
                    summary, batch_loss = self.evaluate(sess, X_batch, Y_batch, merge, training=True)
                    # todo: make visualizable in tensorboard in real time (ideally)
                    # tf.summary.scalar("batch_loss", batch_loss)
                    print("BATCH LOSS {}".format(batch_loss))

                    train_writer.add_summary(summary, counter)
                    self.writer.add_summary(summary, counter)

                    # save the loss for later
                    self.loss_list.append(batch_loss)

                    # plot
                    if batch_idx%10 == 0:
                        total_examples = batch_size * num_batches * epoch_idx + batch_size * batch_idx + batch_size

                        # print stats
                        serialize_to_file(self.loss_list, outfolder=self.log_dir)
                        tf.logging.info("Epoch: {0:3d} | Batch: {1:3d} | TotalExamples: {2:5d} | BatchLoss: {3:8.4f}".format(epoch_idx, batch_idx,
                                                                         total_examples, batch_loss))

                # validate after each epoch
                summary, validation_loss = self.evaluate(sess, X_val, Y_val, merge)

                tf.summary.scalar("validation_loss", validation_loss)

                train_writer.add_summary(summary, counter)
                self.writer.add_summary(summary, counter)
                mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
                tf.logging.info("Epoch Over: {0:3d} | MeanEpochLoss: {1:8.4f} | ValidationSetLoss: {2:8.4f} \n".format(epoch_idx, mean_epoch_loss, validation_loss))
                # todo: write validation loss to tensorboard feed

        self.save_training_time()

    def test(self, dataset):
        # , batch_size, epochs):
        # session
        with tf.Session() as sess:
            # initialize global vars
            sess.run(tf.global_variables_initializer())


            # tf.summary.scalar("total_loss", self.total_loss)
            # validation_loss = 0
            # tf.summary.scalar("validation_loss", validation_loss)

            test_writer = tf.summary.FileWriter('{}/test'.format(self.log_output), sess.graph)

            # fetch validation and test sets
            X_test, Y_test = dataset.get_test_data()

            merge=tf.summary.merge_all()

            summary, test_loss = self.evaluate(sess, X_test, Y_test, merge)

            # todo: this is writing to incorrect variable
            tf.summary.scalar("test_loss", test_loss)

            test_writer.add_summary(summary)
            print("Test set loss:", test_loss)

    def evaluate(self, sess, X, Y, merge=None, training=False):

        # fill (X,Y) placeholders
        feed_dict = {self.input_x: X, self.input_y: Y}
        batch_size = X.shape[0]

        # fill initial state
        for init_state in self.init_states:
            # init_state: [batch_size x cell.state_size[i]]
            feed_dict[init_state] = np.random.uniform(-self.init_state_C, self.init_state_C, [batch_size, init_state.shape[1]])

        # run and return the loss
        if training:
            if merge is None:
                tf.log.ERROR("No merge included - summary cannot be written")
                loss, _ = sess.run([self.total_loss, self.train_step], feed_dict)
            else:
                summary, loss, _ = sess.run([merge, self.total_loss, self.train_step], feed_dict)
                return summary, loss
        else:
            summary, loss = sess.run([merge, self.total_loss], feed_dict)
            print(summary)
            print(loss)
            return summary, loss

        return loss

    # loss list getter
    def get_loss_list(self):
        return self.loss_list

    # save trial info to file
    def save_output_info(self):

        with open(os.path.join(self.log_dir, "info.txt"), "w") as outfile:
            json.dump(self.output_info, outfile)

    # add info about how long trial took to train
    def save_training_time(self):

        with open(os.path.join(self.log_dir, "info.txt"), "w") as json_file:
            duration = datetime.now() - self.start_time
            self.output_info["end_time"] = datetime.now().strftime("%X")
            self.output_info["training_time"] = duration.total_seconds()
            json.dump(self.output_info, json_file)
