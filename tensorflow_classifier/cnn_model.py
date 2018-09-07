# Imports
import numpy as np
import tensorflow as tf

def simplified_cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    batch_size = -1  # batch dimension is dynamically computed
    image_width = 200
    image_height = 200
    channels = 3


    input_layer = tf.reshape(
            features,
            [batch_size, image_width, image_height, channels])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool1_flat = tf.reshape(pool1, [-1, 100 * 100 * 8])
    dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # Our final layer has 2 units, representing the two final classes
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = _predictions(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    batch_size = -1  # batch dimension is dynamically computed
    image_width = 400
    image_height = 400
    channels = 3

    input_layer = tf.reshape(
            features,
            [batch_size, image_width, image_height, channels])

    # Convolutional Layer #1
    norm1 = tf.layers.batch_normalization(input_layer)
    conv1 = tf.layers.conv2d(
        inputs=norm1,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    norm2 = tf.layers.batch_normalization(pool1)
    conv2 = tf.layers.conv2d(
        inputs=norm2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)


    norm22 = tf.layers.batch_normalization(conv2)
    conv22 = tf.layers.conv2d(
        inputs=norm22,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv22, pool_size=[2, 2], strides=2)

    norm3 = tf.layers.batch_normalization(pool2)
    conv3 = tf.layers.conv2d(
        inputs=norm3,
        filters=64,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu)

    norm4 = tf.layers.batch_normalization(conv3)
    conv4 = tf.layers.conv2d(
        inputs=norm4,
        filters=64,
        kernel_size=[7, 7],
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    norm5 = tf.layers.batch_normalization(pool4)
    conv5 = tf.layers.conv2d(
        inputs=norm5,
        filters=64,
        kernel_size=[9, 9],
        padding="same",
        activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool5, [-1, 25 * 25 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # Our final layer has 2 units, representing the two final classes
    logits = tf.layers.dense(inputs=dropout, units=2)
    # logits = tf.nn.softmax(dropout)
    #class_weights = tf.constant([0.28, 0.72])
    #scaled_logits = tf.multiply(logits, class_weights)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    predictions = _predictions(onehot_labels, logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)

    #weights = 1.8
    #loss = tf.nn.weighted_cross_entropy_with_logits(
    #    targets=onehot_labels, logits=logits)

    #loss = tf.reduce_mean(loss)
    #loss = tf.losses.softmax_cross_entropy(
    #    onehot_labels=onehot_labels, logits=logits)

    #loss = tf.losses.log_loss(
#	    onehot_labels,
#	    logits)
    #loss = tf.losses.softmax_cross_entropy(
    #    onehot_labels=onehot_labels, logits=logits)
    #loss = tf.nn.softmax_cross_entropy_with_logits(logits = scaled_logits, labels = onehot_labels)
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=scaled_logits)
    loss = tf.nn.weighted_cross_entropy_with_logits(
      onehot_labels,
      logits,
      0.05,
      name=None)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', loss)

    merged = tf.summary.merge_all()

    #Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _predictions(labels, output_layer):
    return {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=output_layer, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(output_layer, name="softmax_tensor"),
        "labels": labels
    }
