import tensorflow as tf
import argparse


# def test_network(self, net, dataset):
#     tf.logging.info('Testing network ', net.name)
#     net.test(dataset)
#     tf.logging.info('Testing network ', net.name, ' done.')
#
# def test_networks(self, adding_problem, memory_problem, urnn=True, lstm=False, timesteps_idx=1):
#     tf.logging.info('Starting testing...')
#     self.test_network(self.ap_urnn, self.ap_data)


def test():
    tf.reset_default_graph()

    # # Add ops to save and restore all the variables.
    # saver = tf.train.Saver()
    #
    # # Later, launch the model, use the saver to restore variables from disk, and
    # # do some work with the model.
    # with tf.Session() as sess:
    #     # Restore variables from disk.
    #     saver.restore(sess, "./tmp/model_ap_urnn.ckpt")
    #     print("Model restored.")

    export_path = "./tmp/model_ap_urnn.ckpt"

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], export_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", '--adding-problem', dest="adding_problem", action='store_true')
    parser.add_argument("-m", '--memory-problem', dest="memory_problem", action='store_true')

    args = parser.parse_args()

    test()