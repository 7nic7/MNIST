import tensorflow as tf
from keras.datasets.mnist import load_data
from keras.utils.np_utils import to_categorical
import numpy as np

n_step = 28
hidden_units = 100
n_input = 28
cell_units = 128
batch_size = 256
n_class = 10

(x_train,y_train),(x_test,y_test) = load_data()
x_train = x_train/255
x_test = x_test/255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(type(x_train[0:10]))
print(x_train[0:128].shape)

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)


with tf.name_scope('inputs_hidden'):
    inputs_input = tf.reshape(x, [-1, n_input])
    inputs_w = tf.Variable(tf.truncated_normal([n_input, hidden_units], stddev=0.1),
                           dtype=tf.float32, name='inputs_w')
    inputs_b = tf.Variable(tf.zeros([hidden_units]), dtype=tf.float32, name='inputs_b')
    inputs_output = tf.matmul(inputs_input, inputs_w) + inputs_b

with tf.name_scope('rnn_cell'):
    cell_input = tf.reshape(inputs_output, [-1, n_step, hidden_units])
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=cell_units)
    initial_state = cell.zero_state(batch_size, tf.float32)
    output, state = tf.nn.dynamic_rnn(cell, cell_input, initial_state=initial_state)

with tf.name_scope('outputs_hidden'):
    outputs_input = state[1]
    outputs_w = tf.Variable(tf.truncated_normal([cell_units, n_class], stddev=0.1),
                            dtype=tf.float32, name='outputs_w')
    outputs_b = tf.Variable(tf.zeros([n_class]), dtype=tf.float32, name='outputs_b')
    outputs_z = tf.matmul(outputs_input, outputs_w) + outputs_b
    outputs = tf.nn.softmax(outputs_z)

with tf.name_scope('train'):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs_z,labels=y)
    step = tf.train.AdamOptimizer(0.0001).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(outputs,1)),tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20):
    for i,j in zip(np.arange(0,x_train.shape[0],batch_size),np.arange(batch_size,x_train.shape[0]-batch_size,batch_size)):
        batch_x = x_train[i:j]
        batch_y = y_train[i:j]
        print(batch_x.shape)
        _,acc = sess.run([step,accuracy], feed_dict={x:batch_x,y:batch_y})
        print(acc)

test_acc = sess.run(accuracy,feed_dict={x:x_test,y:y_test})
print(test_acc)
















