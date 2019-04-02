import tensorflow.examples.tutorials.mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(r"G:\python_file\mnist", one_hot=True)
EPOCH = 1000
BATCH_SIZE = 64
NOISE_DIM = 50


def generator(noise, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        hidden_g = tf.layers.dense(noise, 128, activation=tf.nn.leaky_relu)
        hidden1_g = tf.layers.dropout(hidden_g, 0.2)
        g_logits = tf.layers.dense(hidden1_g, 784)
        g_out = tf.nn.tanh(g_logits)
    return g_out


def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        hidden_d = tf.layers.dense(inputs, 128, activation=tf.nn.leaky_relu)
        d_logits = tf.layers.dense(hidden_d, 1)
        d_out = tf.nn.sigmoid(d_logits)
        return d_logits, d_out


noise = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NOISE_DIM])
# noise2 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, NOISE_DIM])
img = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 784])
with tf.name_scope('update_D'):
    g_out = generator(noise, reuse=False)
    d_logits, d_out = discriminator(img, reuse=False)
    d_logits_reuse, d_out_reuse = discriminator(g_out, reuse=True)


with tf.name_scope('update_G'):
    g_out2 = generator(noise, reuse=True)
    d_logits_reuse2, d_out_reuse2 = discriminator(g_out2, reuse=True)


with tf.name_scope('optimizer'):
    # 下面两行可以用来固定不想训练的参数
    var_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    var_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
    loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.ones([BATCH_SIZE, 1], dtype='float32')*0.9,
                                                                         logits=d_logits))
    loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.zeros([BATCH_SIZE, 1], dtype='float32')*0.9,
                                                                         logits=d_logits_reuse))
    loss_d = loss_d_fake+loss_d_real
    op_d = tf.train.AdamOptimizer(0.0001).minimize(loss_d, var_list=var_d)

    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=np.ones([BATCH_SIZE, 1], dtype='float32')*0.9,
                                                                    logits=d_logits_reuse2))
    op_g = tf.train.AdamOptimizer(0.0001).minimize(loss_g, var_list=var_g)


# real_labels = np.zeros([BATCH_SIZE, 2])
# real_labels[:, 0] = 1
#
# fake_labels = np.zeros([BATCH_SIZE, 2])
# fake_labels[:, 1] = 1

sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_ds = []
loss_gs = []
# samples = []
# index = list(range(2*BATCH_SIZE))
f, a = plt.subplots(1, 5)
plt.ion()

for i in range(EPOCH):
    # print(tf.trainable_variables())
    for j in range(mnist.train.images.shape[0]//BATCH_SIZE):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        batch_xs = batch_xs * 2 - 1
        noise_value = np.random.uniform(-1, 1, (BATCH_SIZE, NOISE_DIM))
        noise_value2 = np.random.uniform(-1, 1, (BATCH_SIZE, NOISE_DIM))
        # 生成图片
        # img_fake = sess.run(g_out, feed_dict={noise: noise_value})
        # 训练 discriminator
        # images_ = np.concatenate([batch_xs, img_fake], axis=0)
        # labels_ = np.concatenate([real_labels, fake_labels], axis=0)
        # np.random.shuffle(index)
        _, loss_d_value = sess.run([op_d, loss_d], feed_dict={img: batch_xs, noise: noise_value})
        # 训练generator
        _, loss_g_value = sess.run([op_g, loss_g], feed_dict={noise: noise_value2})
    noise_value3 = np.random.uniform(-1, 1, (BATCH_SIZE, NOISE_DIM))
    # loss_d_value = sess.run(loss_d, feed_dict={img: batch_xs, noise: noise_value3})
    # loss_g_value = sess.run(loss_g, feed_dict={noise: noise_value3})
    # loss_ds.append(loss_d_value)
    # loss_gs.append(loss_g_value)

    sample_img = sess.run(g_out, feed_dict={noise: noise_value3})
    for image_i in range(5):
        a[image_i].clear()
        a[image_i].imshow(sample_img[image_i].reshape([28, 28]), cmap='gray')
    plt.title(i+1)
    plt.draw(); plt.pause(0.1)
    # samples.append(sample_img)
    print('%s epoch: discriminator loss : %s ; generator loss is %s' % (i, loss_d_value, loss_g_value))
plt.ioff()
plt.show()
