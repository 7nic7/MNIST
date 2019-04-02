import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as inputdata
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
# now,test accuracy: 96.45%
mnist = inputdata.read_data_sets(r"G:\python file", one_hot=True)
print('done')
with tf.name_scope('inputs'):
    x = tf.placeholder(dtype=tf.float32, name='x_input')
    y_ = tf.placeholder(dtype=tf.float32, name='y_input')
#conv1
with tf.name_scope('conv1'):
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1),
                          dtype=tf.float32, name='weights_conv1')
    B_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name='biases_conv1')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    activation_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, padding='SAME', strides=[1, 1, 1, 1]) + B_conv1)
#pool1
with tf.name_scope('pool1'):
    pool1 = tf.nn.max_pool(activation_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#conv2
with tf.name_scope('conv2'):
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1),
                          dtype=tf.float32, name='weights_conv2')
    B_conv2 = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), name='biaese_conv2')
    activation_conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, padding='SAME', strides=[1, 1, 1, 1]) + B_conv2)
#pool2
with tf.name_scope('pool2'):
    pool2 = tf.nn.max_pool(activation_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#flatten
with tf.name_scope('flatten'):
    flatten = tf.reshape(pool2, shape=[-1, 7*7*64])
#fc1
with tf.name_scope('fc1'):
    W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1), dtype=tf.float32, name='weights_fc1')
    B_fc1 = tf.Variable(tf.constant(0.1, shape=[1024], dtype=tf.float32), name='biaese_fc1')
    fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + B_fc1)
#propout
    keep_prop = tf.placeholder(dtype=tf.float32)
    drop = tf.nn.dropout(fc1, keep_prop)
#fc2
with tf.name_scope('outputs'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), dtype=tf.float32)
    B_fc2 = tf.Variable(tf.constant(0.1, shape=[10], dtype=tf.float32))
    outputs = tf.matmul(drop, W_fc2) + B_fc2
    fc2 = tf.nn.softmax(outputs)
#train
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs,labels=y_))
    tf.summary.scalar('loss', cost)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(fc2, 1)), 'float'))
tf.summary.scalar('accuracy', accuracy)

sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('G:/python file/logs/train', sess.graph)
test_writer = tf.summary.FileWriter('G:/python file/logs/test', sess.graph)
init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(32)
    sess.run(train_step, feed_dict={x:batch_x, y_:batch_y, keep_prop:0.5})
    if i%100 == 0:
        train_result = sess.run(merged, feed_dict={x: batch_x, y_: batch_y, keep_prop: 1.0})
        train_writer.add_summary(train_result, i)
        test_result = sess.run(merged, feed_dict={x:mnist.test.images[0:1000],
                                                  y_: mnist.test.labels[0:1000], keep_prop:1.0})
        test_writer.add_summary(test_result, i)
print('done')

# l = np.argmax(mnist.validation.labels, axis=1)
# images = mnist.validation.images[l==0]
# print(images.shape)
feed_dict = {x:mnist.train.images[0], keep_prop:1.0,y_:mnist.train.labels[0]}
# g1 = tf.gradients(ys=pool1, xs=x)
# g1_value = sess.run(g1,feed_dict=feed_dict)
# g2 = tf.gradients(ys=pool2, xs=x)
# g2_value = sess.run(g2,feed_dict=feed_dict)
g3 = tf.gradients(ys=outputs[0,np.argmax(mnist.train.labels[0])], xs=x)
g3_value = sess.run(g3, feed_dict=feed_dict)[0]
# g3_value = np.maximum(g3_value, 0)
v = preprocessing.minmax_scale(g3_value)
v = np.clip(v*255.0, 0.0, 255.0).astype(np.uint8)

plt.figure()
plt.imshow(g3_value.reshape([28,28]), cmap='jet')
# plt.figure()
# g12_value = (g1_value[0] + g2_value[0]) /2
# plt.imshow(g12_value.reshape([28,28]), cmap='gray')
#
#
# plt.figure()
# g123_value = (g1_value[0] + g2_value[0] + g3_value[0]) / 2
# plt.imshow(g123_value.reshape([28,28]), cmap='gray')
# plt.show()
# import tensorflow as tf
# import numpy as np
# X = np.random.rand(100).reshape([-1,1])
# Y = 0.1*X + 0.3
#
# #baiese = 0.3, Weights = 0.1
#
# def add_layer(inputs, input_dim, out_dim, activation_function=None):
#     Weights = tf.Variable(tf.truncated_normal(shape=[input_dim, out_dim], stddev=0.1, dtype=tf.float32))
#     baies = tf.Variable(tf.zeros([out_dim], dtype=tf.float32) + 0.1)
#     z = tf.matmul(inputs, Weights) + baies
#     if activation_function is None:
#         outputs = z
#     else:
#         outputs = activation_function(z)
#     return outputs
#
# x = tf.placeholder(dtype=tf.float32)
# y = tf.placeholder(dtype=tf.float32)
# # outputs = add_layer(x, 1, 1, activation_function=tf.nn.relu)
# # cost = tf.reduce_mean(tf.square(outputs-y))
# # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
# # init_op = tf.global_variables_initializer()
# # sess = tf.Session()
# # saver = tf.train.Saver()
# # sess.run(init_op)
# # for i in range(1000):
# #     sess.run(train_step, feed_dict={x: X, y: Y})
# #     if i%100 == 0:
# #         print(sess.run(cost, feed_dict={x:X, y:Y}))
# sess = tf.Session()
# saver = tf.train.Saver()
# saver.restore(sess, r'C:\Users\tianping\Desktop\model')
# print(sess.run(cost, feed_dict={x:X, y:Y}))

##不要规定占位符的shape， 这样feed_dict里面就可以随意填入我们想训练的数据了