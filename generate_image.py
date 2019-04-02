import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as inputdata
import numpy as np
import matplotlib.pyplot as plt
# now,test accuracy: 96.45%
data = np.load('C:/Users/tianping/Desktop/data.npy')
dic = data.item()

# mnist = inputdata.read_data_sets(r"G:\python file", one_hot=True)
with tf.name_scope('inputs'):
    x = tf.Variable(np.zeros([28, 28, 1]), dtype=tf.float32, name='inputs')
    labda = tf.placeholder(dtype=tf.float32, name='lambda')
    # y_ = tf.placeholder(dtype=tf.float32, name='y_input')
#conv1
with tf.name_scope('conv1'):
    W_conv1 = tf.Variable(dic['conv1/weights_conv1:0'],trainable=False,
                          dtype=tf.float32, name='weights_conv1')
    B_conv1 = tf.Variable(dic['conv1/biases_conv1:0'],trainable=False,
                          dtype=tf.float32, name='biases_conv1')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    activation_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, padding='SAME', strides=[1, 1, 1, 1]) + B_conv1)
#pool1
with tf.name_scope('pool1'):
    pool1 = tf.nn.max_pool(activation_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#conv2
with tf.name_scope('conv2'):
    W_conv2 = tf.Variable(dic['conv2/weights_conv2:0'],trainable=False,
                          dtype=tf.float32, name='weights_conv2')
    B_conv2 = tf.Variable(dic['conv2/biaese_conv2:0'], trainable=False,
                          name='biaese_conv2')
    activation_conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, padding='SAME', strides=[1, 1, 1, 1]) + B_conv2)
#pool2
with tf.name_scope('pool2'):
    pool2 = tf.nn.max_pool(activation_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#flatten
with tf.name_scope('flatten'):
    flatten = tf.reshape(pool2, shape=[-1, 7*7*64])
#fc1
with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(dic['fc1/weights_fc1:0'],trainable=False,
                        dtype=tf.float32, name='weights_fc1')
    B_fc1 = tf.Variable(dic['fc1/biaese_fc1:0'],trainable=False,dtype=tf.float32,
                        name='biaese_fc1')
    fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + B_fc1)
#propout
    keep_prop = tf.placeholder(dtype=tf.float32)
    drop = tf.nn.dropout(fc1, keep_prop)
#fc2
with tf.name_scope('outputs'):
    W_fc2 = tf.Variable(dic['outputs/Variable:0'],trainable=False, dtype=tf.float32)
    B_fc2 = tf.Variable(dic['outputs/Variable_1:0'],trainable=False)
    outputs = tf.matmul(drop, W_fc2) + B_fc2
    fc2 = tf.nn.softmax(outputs)
#train
with tf.name_scope('loss'):
    print(outputs.get_shape())
    cost = -outputs[0,0] + labda * tf.reduce_sum(tf.multiply(x,x))
    tf.summary.scalar('loss', cost)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(fc2, 1)), 'float'))
# tf.summary.scalar('accuracy', accuracy)

sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('G:/python file/logs/train', sess.graph)
# test_writer = tf.summary.FileWriter('G:/python file/logs/test', sess.graph)
init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(1000):
    # batch_x, batch_y = mnist.train.next_batch(32)
    _,train_result = sess.run([train_step,merged], feed_dict={labda:100, keep_prop:1.0})
    if i%100 == 0:
    #     train_result = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prop: 1.0})
        train_writer.add_summary(train_result, i)
    #     test_result = sess.run(merged, feed_dict={x:mnist.test.images[0:2000],
    #                                               y_: mnist.test.labels[0:2000], keep_prop:1.0})
    #     test_writer.add_summary(test_result, i)
print('done')

# class model visualisation
x = sess.run(x)
plt.imshow(x[:,:,0], cmap='gray')
plt.show()

# class saliency visualisation
