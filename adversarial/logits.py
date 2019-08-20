import numpy as np
import tensorflow as tf
import os
import json


class CheckpointInference:
    def __init__(self, folder, checkpoint_number, session=None):

        # read in dictionary information about the run for later use
        with open(os.path.join(folder, "info.txt")) as f:
            run_info = json.load(f)

        # construct full path to checkpoint
        checkpoint_name = "model_mnist_{}.ckpt-{}.meta".format(run_info["net_type"], checkpoint_number)
        # check if file exists
        full_path_to_checkpoint = os.path.join(folder, checkpoint_name)
        assert os.path.isfile(full_path_to_checkpoint), "Checkpoint does not exist."

        # set up graph
        # tf.reset_default_graph()
        graph = tf.get_default_graph()

        # start new session
        if session is None:
            self.sess = tf.compat.v1.Session()
        else:
            self.sess = session

        # create saver and restore
        saver = tf.compat.v1.train.import_meta_graph(full_path_to_checkpoint)
        # restore graph from checkpoint
        saver.restore(self.sess, full_path_to_checkpoint[:-5])

        # get relevant tensor hooks from graph
        self.logits = graph.get_tensor_by_name("Identity:0")  # logits tensor (outputs_o)
        self.init_state = graph.get_tensor_by_name("init_state:0")  # initial state tensor (for creating dictionary)
        input_name = "input_x_mnist_{}:0".format(run_info["net_type"])
        self.input_x = graph.get_tensor_by_name(input_name)  # get input tensor (for feeding in serialized image tensor)

        # initial state
        self.init_state_C = np.sqrt(3 / (2 * run_info["num_hidden"]))

    def get_logits(self, input_tensor):

        # batch_size = int(input_tensor.shape[0])  # previously len(input_tensor)
        #
        # init_state_value = np.random.uniform(-self.init_state_C,
        #                                      self.init_state_C,
        #                                      [batch_size, self.init_state.shape[1]])
        #
        # # convert from tensor to np array
        # input_tensor = input_tensor.eval(session=self.sess)
        # feed_dict = {self.input_x: input_tensor, self.init_state: init_state_value}
        #
        # # COMPUTE THE LOGITS
        # logits = self.logits.eval(session=self.sess, feed_dict=feed_dict)

        if type(input_tensor) == np.array:
            logits = self.get_logits_from_array(input_tensor)
        elif type(input_tensor) == tf.Tensor:
            input_array = input_tensor.eval(session=self.sess)
            logits = self.get_logits_from_array(input_array)
        else:
            tf.logging.error("Input is wrong form.")

        return logits


    def get_logits_from_array(self, input_array):

        batch_size = len(input_array)

        init_state_value = np.random.uniform(-self.init_state_C,
                                             self.init_state_C,
                                             [batch_size, self.init_state.shape[1]])
        # specify dictionary to feed tensorflow placeholder values
        feed_dict = {self.input_x: input_array, self.init_state: init_state_value}

        # COMPUTE THE LOGITS
        logits = self.logits.eval(session=self.sess, feed_dict=feed_dict)

        return tf.convert_to_tensor(logits)

    def get_probabilities(self, input_tensor):

        logits = self.get_logits(input_tensor)
        probabilities = tf.nn.softmax(logits)

        return probabilities

    def get_predictions_from_logits(self, logits):

        predictions = tf.nn.softmax(logits)
        predicted_classes = np.argmax(np.array(predictions.eval(session=self.sess)), axis=1)

        return predicted_classes

    def get_predictions(self, input_tensor):

        logits = self.get_logits(input_tensor)
        predicted_classes = self.get_predictions_from_logits(logits)

        return predicted_classes
