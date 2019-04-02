import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist
import matplotlib.pyplot as plt

mnist = tf.examples.tutorials.mnist.input_data.read_data_sets(r"G:\python file", one_hot=True)

BATCH_SIZE = 100
EPOCH = 600


x = tf.placeholder(shape=[BATCH_SIZE, 784], dtype=tf.float32)
y = tf.placeholder(shape=[BATCH_SIZE, 10], dtype=tf.float32)
z = tf.placeholder(shape=[BATCH_SIZE, 100], dtype=tf.float32)

with tf.variable_scope('generator'):
    inputs_g = tf.concat([z, y], axis=1)                    # different from gan
    hidden_g = tf.layers.dense(inputs_g, 128, tf.nn.relu)
    hidden_g = tf.nn.dropout(hidden_g, 0.5)
    outputs_g = tf.layers.dense(hidden_g, 784, tf.nn.tanh)


def discriminator(image, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        inputs_d = tf.concat([image, y], axis=1)            # different from gan
        hidden_d = tf.layers.dense(inputs_d, 128, tf.nn.leaky_relu)
        hidden_d = tf.nn.dropout(hidden_d, 0.5)
        logits = tf.layers.dense(hidden_d, 1)
        outputs_d = tf.nn.sigmoid(logits)
    return logits, outputs_d


logits_real, outputs_real = discriminator(x, reuse=False)
logits_fake, outputs_fake = discriminator(outputs_g, reuse=True)

var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
with tf.variable_scope('optimizer'):
    # 训练集中没有 真实images但是标签错误的 data
    loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,
                                                                         labels=np.ones([BATCH_SIZE, 1], np.float32)))
    loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                                         labels=np.zeros([BATCH_SIZE, 1], np.float32)))
    loss_d = loss_d_real + loss_d_fake

    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                                    labels=np.ones([BATCH_SIZE, 1], np.float32)))

    op_d = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_d, var_list=var_d)
    op_g = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_g, var_list=var_g)


num_batch = mnist.train.images.shape[0] // BATCH_SIZE // (3+1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

samples = []
loss_ds = []
loss_gs = []

for i in range(EPOCH):
    for _ in range(num_batch):
        for d_iter in range(5):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            batch_zs = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
            _ = sess.run(op_d, feed_dict={x: batch_xs*2-1, y: batch_ys, z: batch_zs})

        _, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        batch_zs = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])

        _ = sess.run(op_g, feed_dict={z: batch_zs, y: batch_ys})

    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    batch_zs = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
    loss_d_value = sess.run(loss_d, feed_dict={x: batch_xs*2-1, y: batch_ys, z: batch_zs})
    loss_g_value, sample_value = sess.run([loss_g, outputs_g], feed_dict={y: batch_ys, z: batch_zs})

    print('epoch %s, discriminator loss is %s, generator loss is %s' % (i, loss_d_value, loss_g_value))

    loss_gs.append(loss_g_value)
    loss_ds.append(loss_d_value)
    samples.append((sample_value, batch_ys))


plt.imshow(samples[-1][0][0].reshape([28,28]), cmap='gray')