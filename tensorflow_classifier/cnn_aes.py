# Imports
import argparse
import os.path
import sys
import time
import numpy as np
import tensorflow as tf
from cnn_model import cnn_model_fn

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_FILE = 'dataset/nature_train_00000-of-00002.tfrecord'
VALIDATION_FILE = 'dataset/nature_validation_00000-of-00002.tfrecord'

FLAGS = None

def read_and_decode(filename_queue):
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)
        })
    # now return the converted data
    label  = tf.cast(features['image/class/label'], tf.int64)
    image  = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
    image  = tf.reshape(image, [200 * 200 * 3, ])
    image  = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    width  = features['image/width']
    height = features['image/height']
    return image, label

def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=100 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=100)

        return images, labels

def train():
    # Load training and eval data
    #with tf.name_scope('input'):
    #   filename_queue = tf.train.string_input_producer(
    #     [TRAIN_FILE], num_epochs=1)
    #image, label = read_and_decode(filename_queue)
    # print(image)
    # returns symbolic label and image

    sess = tf.Session()

    images, labels = inputs(train = True, batch_size = 100, num_epochs = None)

    # Required. See below for explanation
    init = tf.initialize_all_variables()

    # grab examples back.
    # first example from file
    #i = 1
    #for i in range(1, 10):
    #    label_val_1, image_val_1, width_val_1, height_val_1 = sess.run([label, image, width, height])
    #    print(image_val_1.set_shape([200 * 200 * 3]).size)
    #    i = i + 1

    classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, model_dir = "/tmp/estimator")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord = coord)

    try:
        step = 0
        while not step == 1000:
            start_time = time.time()

            loss_value = sess.run([classifier])

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d (%.3f seconds)' % (step, duration))

            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def inputs_fn():
    return inputs(train = True, batch_size = 100, num_epochs = None)


def main(unused_argv):
    classifier = tf.estimator.Estimator(
        model_fn = cnn_model_fn, model_dir = "/tmp/estimator")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    classifier.train(input_fn = lambda: inputs(train = True, batch_size = 100, num_epochs = None), steps = 20000, hooks = [logging_hook])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/data',
        help='Directory with the training data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
