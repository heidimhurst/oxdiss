import numpy as np
import tensorflow as tf
# import kfac

# from stochqn.tf import TensorflowStochQNOptimizer

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


def serialize_to_file(loss, outfolder="", name="losses.txt"):
    file=open(os.path.join(outfolder, name), 'w')
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
        output_info={},
        checkpoints=0,
        resume=""):

        # self
        self.name = name
        self.problem = self.name.split("_")[0]
        self.net_type = self.name.split("_")[1]
        self.loss_list = []
        self.validation_list = []
        self.init_state_C = np.sqrt(3 / (2 * num_hidden))
        # self.log_dir = './logs/{}_{}/logs'.format(self.log_output, self.name)
        top_dir = "./logs/{}/{}/".format(self.problem, self.net_type)

        # if resuming:
        if resume != "":
            # evaluate past
            self.log_dir = "./logs/{}/{}/{}/logs".format(self.problem, self.net_type, resume)
            tf.logging.info("Attempting to resume from previous checkpoint in {}".format(self.log_dir))
            assert os.path.isdir(self.log_dir), "Log directory must exist"
            self.resume = True

        else:
            self.log_dir = "./logs/{}/{}/{}/logs".format(self.problem, self.net_type, increment_trial(top_dir))
            self.resume = False

        self.log_output = self.log_dir[:-5]
        self.writer = tf.summary.FileWriter(self.log_dir)

        # set up checkpoint saving/resume
        self.checkpoints = checkpoints
        # ==== CHECKPOINT SAVING ====
        self.checkpoint_name = os.path.join(self.log_dir, "model_{}.ckpt".format(self.name))
        # create checkpoint saver & manager
        # ckpt = tf.train.Checkpoint(step)
        # manager = tf.train.CheckpointManager(ckpt, self.checkpoint_name, max_to_keep=20)
        # ckpt.restore(manager.lagest_checkpoint)
        # if manager.latest_checkpoint:
        #     tf.logging.info("Restored from {}".format(manager.latest_checkpoint))
        # else:
        #     tf.logging.info("Initializing from scratch.")


        self.validation_frequency = 50

        # init cell
        if isinstance(rnn_cell, str):
            self.cell = FLEXICell(num_units=num_hidden, num_in=num_in, components=rnn_cell)
        elif rnn_cell == (URNNCell or REFLECTCell or RDCell or PRDCell or FRPDICell):
            self.cell = rnn_cell(num_units = num_hidden, num_in = num_in)
        else: # for simple RNN, LSTM, etc
            self.cell = rnn_cell(num_units = num_hidden, activation = activation_hidden, )


        # extract output size
        self.output_size = self.cell.output_size # TWICE number of outputs specified
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

        # TESTING
        # make outputs_o a non-trainable variable
        # outputs_o = tf.Variable(outputs_o, trainable=False, name="logits", validate_shape=False)
        self.outputs_o = outputs_o

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

        # === PREDICTIONS (beta) ===
        if "mnist" in self.problem:
            # predictions = {
            #     "classes": tf.argmax(input=outputs_o, axis=1),
            #     "probabilities": tf.nn.softmax(outputs_o, name="softmax_tensor")
            # }
            # self.predict = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions)

            # self.logits = outputs_o  # TODO: need to somehow save logits for later use/export
            # self.logits = tf.get_variable("logits", initializer=outputs_o, trainable=False)

            self.predict = tf.nn.softmax(outputs_o)
            # create placeholders
            self.true_class_placeholder = tf.placeholder(tf.uint8,
                                                    # shape=(output_info["batch_size"], 1),
                                                    name="Y_batch")
            self.predicted_class_placeholder = tf.placeholder(tf.int64,
                                                         # shape=(output_info["batch_size"]),
                                                         name="predicted_classes")
            self.accuracy, self.acc_opp = tf.metrics.accuracy(labels=self.true_class_placeholder,
                                                              predictions=self.predicted_class_placeholder,
                                                              name="accuracy_metric")

        # === KFAC Optimization (beta) ===
        if optimizer == "adaqn":
            # optimizer = TensorflowStochQNOptimizer(self.total_loss, optimizer="adaQN")
            # self.train_step = optimizer.minimize(self.total_loss)
            tf.logging.warn("Not implemented for optimizer adaqn.")
        elif optimizer != "kfac":
            self.train_step = optimizer.minimize(self.total_loss, name='Optimizer')
        # elif optimizer == "kfac":
        #     tf.logging.info("Initializing KFAC situation")
        #     # register loss
        #     layer_collection = kfac.LayerCollection()
        #     layer_collection.register_softmax_cross_entropy_loss(logits=outputs_o)
        #     # register layers
        #     layer_collection.auto_register_layers(batch_size=128)
        #     # construct training op
        #     optimizer = kfac.PeriodicInvCovUpdateKfacOpt(learning_rate=0.0001,
        #                                                  damping=0.001,
        #                                                   momentum=0.9,
        #                                                   cov_ema_decay=0.95,
        #                                                   invert_every=10,
        #                                                   cov_update_every=1,
        #                                                  layer_collection=layer_collection)
        #     self.train_step = optimizer.minimize(self.total_loss)

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

        tf.logging.info('Network __init__ over. Number of trainable params={}'.format(t_params))

        # output info for later use
        self.start_time = datetime.now()
        self.output_info = output_info
        self.output_info.update({"name": self.name, "net_type": self.net_type, "problem":self.problem,
                                 "directory": self.log_dir, "optimizer": optimizer.get_name(),
                                 "num_in": num_in, "num_out": num_out, "num_hidden": num_hidden,
                                 "num_target": num_target, "cell_type": self.cell._base_name,
                                 "trainable_params": int(t_params), "validation_frequency": self.validation_frequency,
                                 "gpu":tf.test.is_gpu_available()})

        # save info about trial run for later use/analysis/recordkeeping
        self.save_output_info()

    @classmethod
    # https://stavshamir.github.io/python/2018/05/26/overloading-constructors-in-python.html
    def from_checkpoint(cls, checkpoint) -> 'TFRNN':
        """
        Creates instance of TFRNN model from checkpoint
        :param checkpoint: saved model checkpoint (should be in a /logs/ folder)
        :return:
        """

        # if checkpoint is a folder
        if os.path.isdir(checkpoint):
            # check if
            pass


    def train(self, dataset, batch_size, epochs):

        # create saver object
        saver = tf.train.Saver()


        # VALTEST
        write_op = tf.summary.merge_all()

        # session
        with tf.Session() as sess:

            # if restore=True, resume from previous (NB: this will continue training/validation on DIFFERENT DATA)
            # where to restore from?
            if self.resume:
                # resume_path = os.path.join(self.checkpoint_name, "logs")
                resume_path = self.log_dir
                tf.logging.info("Attempting to restore from {}".format(resume_path))
                saver.restore(sess, tf.train.latest_checkpoint(resume_path))
                # how to reset counter to resume from previous location?
                counter = self.get_step_from_checkpoint() + 1
                tf.logging.info("Resuming training for {} at step {}".format(self.name, counter))
            else:
                counter = 0
                tf.logging.info("Starting training for {}".format(self.name))


            # print(self.w_ho.eval())
            # initialize loss
            # batch_loss = tf.Variable(0., name="batch_loss")
            tf.summary.scalar("total_loss", self.total_loss)

            # ACCURACY
            if "mnist" in self.problem:
                tf.summary.scalar("accuracy", self.accuracy)
            # summary, validation_loss = self.evaluate(sess, X_val, Y_val, merge)


            # TODO: write validation loss os single tensor : HOW :(
            # how to write validation loss? need to initialize tensor correctly, at present just keeps
            # getting written as nans, or however i set this initial value - does it need to be a function? :(

            # self.validation_loss = tf.placeholder(tf.float32, name="validation_loss_" + self.name)
            # self.validation_loss = tf.Variable(np.nan)

            # VALTEST
            # validation_loss = tf.Variable(np.nan)
            # tf.summary.scalar("validation_loss", validation_loss)

            # BATCHTEST
            # batch_loss = tf.Variable(np.nan)
            # tf.summary.scalar("batch_loss", batch_loss)

            train_writer = tf.summary.FileWriter('{}/train'.format(self.log_output), sess.graph)
            eval_writer = tf.summary.FileWriter('{}/eval'.format(self.log_output), sess.graph)

            # train_writer = tf.contrib.summary('./logs/{}_{}/train'.format(self.log_output, self.name), sess.graph)

            # ACCURACY PLACEHOLDER


            # initialize global vars
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            stream_vars = [i for i in tf.local_variables()]
            print(stream_vars)

            # fetch validation and test sets
            num_batches = dataset.get_batch_count(batch_size)
            X_val, Y_val = dataset.get_validation_data()

            # init loss list
            self.loss_list = []
            self.validation_list = []
            tf.logging.info("NumEpochs: {0:3d} |  BatchSize: {1:3d} |  NumBatches: {2:5d} \n".format(epochs,
                                                                                              batch_size,
                                                                                              num_batches))
            # train for several epochs
            for epoch_idx in range(epochs):

                epoch_start = datetime.now()

                tf.logging.info("Epoch Starting:{} \n ".format(epoch_idx))

                # time for batch start
                batch_start = datetime.now()

                # train on several minibatches
                # for batch_idx in range(num_batches):
                for batch_idx in range(3):
                    counter += 1  # for use with summary writer
                    merge = tf.summary.merge_all()

                    # save every n steps
                    if self.checkpoints > 0:
                        if counter % self.checkpoints == 0:
                            save_path = saver.save(sess, os.path.join(self.log_dir, "model_{}.ckpt".format(self.name)),
                                                   global_step=counter)
                            print("Model checkpoint saved in path: %s" % save_path)

                    # get one batch of data
                    # X_batch: [batch_size x time x num_in]
                    # Y_batch: [batch_size x time x num_target] or [batch_size x num_target] (single_output?)
                    X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)

                    # evaluate
                    summary, batch_loss, *prediction = self.evaluate(sess, X_batch, Y_batch, merge, training=True)
                    # summary, batch_loss, *prediction = self.evaluate(sess, X_val, Y_val, merge, training=True)
                    # tf.summary.scalar("batch_loss", batch_loss)
                    print("BATCH LOSS {}".format(batch_loss))

                    # if MNIST, get accuracy
                    if "mnist" in self.problem:
                        self.get_accuracy(sess, prediction=prediction, Y_batch=Y_batch)


                    # BATCHTEST
                    # summary = sess.run(write_op, {batch_loss: batch_loss})  # NOT SURE IF THIS WORKS
                    train_writer.add_summary(summary, counter)
                    train_writer.flush()  # NOT SURE IF THIS WORKS

                    self.writer.add_summary(summary, counter)

                    # save the loss for later
                    self.loss_list.append(batch_loss)

                    # print information every 10 batches
                    if batch_idx % 10 == 0:
                        total_examples = batch_size * num_batches * epoch_idx + batch_size * batch_idx + batch_size

                        # print stats
                        serialize_to_file(self.loss_list, outfolder=self.log_dir)

                        avg_batch_duration = datetime.now() - batch_start
                        avg_batch_duration = (avg_batch_duration.total_seconds())/10.0

                        tf.logging.info("Step: {5} | Epoch: {0:3d} | Batch: {1:3d} | TotalExamples: {2:5d} | BatchLoss: {3:8.4f} | Average Batch Time: {4:8.4f}".format(
                                        epoch_idx, batch_idx, total_examples, batch_loss, avg_batch_duration, counter))

                        batch_start = datetime.now()

                    # validation loss
                    if batch_idx % self.validation_frequency == 0:
                        # TODO: how frequently should we validate?
                        # validate after every 10 batches?

                        if "mnist" in self.problem:
                            summary, validation_loss, prediction, Y = self.evaluate(sess, X_val, Y_val, merge,
                                                                     max_batch_size=50, training=False)
                            self.get_accuracy(sess, [prediction], Y)
                        else:
                            summary, validation_loss = self.evaluate(sess, X_val, Y_val, merge, training=False)

                        self.validation_list.append(validation_loss)
                        serialize_to_file(self.validation_list, outfolder=self.log_dir, name="validation_losses.txt")

                        # write evaluation, close stream
                        eval_writer.add_summary(summary, counter)
                        eval_writer.flush()

                        tf.logging.info("Step: {5} | Epoch: {0:3d} | Batch: {1:3d} | TotalExamples: {2:5d} | BatchLoss: {3:8.4f} | ValidationLoss: {4:8.4f}".format(
                                        epoch_idx, batch_idx,
                                        total_examples, batch_loss, validation_loss, counter))

                # if self.problem == "mnist":
                #     tf.summary.scalar("validation_loss", validation_loss)

                train_writer.add_summary(summary, counter)
                self.writer.add_summary(summary, counter)
                mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
                epoch_duration = datetime.now() - epoch_start
                tf.logging.info("Epoch Over: {0:3d} | MeanEpochLoss: {1:8.4f} | ValidationSetLoss: {2:8.4f} | Time: {3:8.4f} \n".format(epoch_idx, mean_epoch_loss, validation_loss, epoch_duration.total_seconds()))

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
            self.output_info["test_loss"] = test_loss

    def evaluate(self, sess, X, Y, merge=None, training=False, max_batch_size=1024):

        # fill (X,Y) placeholders
        batch_size = X.shape[0]

        # todo: if batch size is too large, SUBSAMPLE and throw warning
        if batch_size > max_batch_size:
            # and self.problem != "mnist":
            tf.logging.warn("Evaluation batch size is too large.  Downsampling from {} to {}.".format(batch_size,
                                                                                                      max_batch_size))
            inds = np.random.randint(0, batch_size-1, max_batch_size)
            X = X[inds]
            Y = Y[inds]

            batch_size = X.shape[0]

        feed_dict = {self.input_x: X, self.input_y: Y}

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
                if "mnist" in self.problem:
                    # PREDICTION
                    summary, loss, prediction, _ = sess.run([merge, self.total_loss,
                                                             self.predict, self.train_step], feed_dict)
                    return summary, loss, prediction
                else:
                    summary, loss, _ = sess.run([merge, self.total_loss,
                                                 self.train_step], feed_dict)
                    return summary, loss
        else:
            # TODO: I'm not entirely sure this is validating correctly -
            # seems like the values are entirely too close, and I'm afraid this is just calling the loss function
            # on the evaluation stuff and thus actually incrementing the model itself? not sure :(
            # NOTE: should be ok because i'm not calling self.train_step :)
            if "mnist" in self.problem:
                summary, loss, prediction = sess.run([merge, self.total_loss, self.predict], feed_dict)
                return summary, loss, prediction, Y
            else:
                summary, loss = sess.run([merge, self.total_loss], feed_dict)
                # tf.logging.info("Evaluating error... loss is {}".format(loss))
                return summary, loss

        return loss

    def get_accuracy(self, sess, prediction, Y_batch):
        predicted_classes = np.argmax(np.array(prediction[0]), axis=1)

        # print(predicted_classes)
        # print(Y_batch)

        # manual comparison (troubleshooting accuracy)
        accuracy_manual = 0
        for guess in range(len(predicted_classes)):
            if predicted_classes[guess] == Y_batch[guess][0]:
                accuracy_manual += 1
        accuracy_manual = accuracy_manual / len(predicted_classes)

        # print("MANUAL ACCURACY:{}".format(accuracy_manual))

        # Next three lines of code prevent accuracy metric from being cumulative
        # (i.e. look at accuracy of each batch SEPARATELY)
        # see info about default behavior: http://ronny.rest/blog/post_2017_09_11_tf_metrics/
        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")
        # Define initializer to initialize/reset running variables
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        sess.run(running_vars_initializer)
        # end undo cumulative observations

        accuracy, acc_opp = sess.run([self.accuracy, self.acc_opp],
                                     feed_dict={self.true_class_placeholder: Y_batch,
                                                self.predicted_class_placeholder: predicted_classes})

        return accuracy, acc_opp

    def get_logits(self, input_tensor, sess=None):
        """
        Callable that takes an input tensor and returns the model logits for use with
        cleverhans implementation of the Fast Gradient Method.
        :param input_tensor: input tensor (test value)
        :return: model logits
        """
        batch_size = 1
        feed_dict = {self.input_x: input_tensor}

        # fill initial state
        for init_state in self.init_states:
            # init_state: [batch_size x cell.state_size[i]]
            feed_dict[init_state] = np.random.uniform(-self.init_state_C,
                                                      self.init_state_C,
                                                      [batch_size, init_state.shape[1]])

        if sess is None:
            sess = tf.compat.v1.Session()

        logits = sess.run([self.outputs_o], feed_dict)

        return logits

    def load_checkpoint(self, checkpoint):
        """
        Loads model from checkpoint filepath (?)
        :param checkpoint:
        :return:
        """

    def attack(self, example):

        assert("mnist" in self.problem)


    # loss list getter
    def get_loss_list(self):
        return self.loss_list

    # log dir getter
    def get_log_dir(self):
        return self.log_dir

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

    # get step number for checkpoint, to use to reiterate
    def get_step_from_checkpoint(self):

        # get checkpoint data
        checkpoints = glob(self.checkpoint_name + "*")

        # cut to just numbers
        steps = [check[check.find("-")+1:] for check in checkpoints]
        steps = [check[:check.find(".")] for check in steps]

        # get largest integer
        step = max([int(step) for step in steps if step.isdigit()])

        return step
