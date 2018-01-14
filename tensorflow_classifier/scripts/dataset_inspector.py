import tensorflow as tf
from google.protobuf.json_format import MessageToJson


sess = tf.InteractiveSession()
for example in tf.python_io.tf_record_iterator("../dataset/nature_train_00000-of-00002.tfrecord"):
    result = tf.train.Example.FromString(example)
    features = tf.parse_single_example(
        example,
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
    #image = features['image/encoded']
    image  = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
    #image  = tf.reshape(image, [200 * 200 * 3, ])
    #image  = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    width  = features['image/width']
    height = features['image/height']


    a = tf.Print(image, [image], message="This is a: ")
    b = tf.add(a, a)

    b.eval()
