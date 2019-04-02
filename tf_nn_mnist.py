import tensorflow.examples.tutorials.mnist
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import cv2

mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets(r"G:\python_file_mnist", one_hot=True)
# 可视化
show_num = 2500
embedding_var = tf.Variable(tf.stack(mnist.test.images[:show_num]), trainable=False, name='embedding_var')

with tf.name_scope('inputs'):
    x = tf.placeholder(shape = [None, 784], dtype=tf.float32, name='x_input')
    y_ = tf.placeholder(shape = [None, 10], dtype=tf.float32, name='y_input')
# 对输入image的可视化
with tf.name_scope('show_images'):
    reshaped_x = tf.reshape(x, [-1, 28, 28, 1], 'reshaped_x')
    tf.summary.image('image', reshaped_x, 16)

with tf.name_scope('outputs'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name='W')
        tf.summary.histogram('outputs/weights', W)
    with tf.name_scope('biases'):
        B = tf.Variable(tf.zeros([10]), dtype=tf.float32, name='b')
        tf.summary.histogram('outputs/biaese', B)
    z = tf.matmul(x, W) + B
    y = tf.nn.softmax(z)

with tf.name_scope('loss'):
    cost = -tf.reduce_mean(y_*tf.log(y))
    tf.summary.scalar(name='loss', tensor=cost)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
tf.summary.scalar('accuracy', accuracy)

sess = tf.Session()
# 创建embedding_var的标签文件（可视化）


def create_metadata_file(file, labels):
    with open(file, 'w') as f:
        f.write('Index' + '\t' + 'Label' + '\n')
        for i in range(show_num):
            f.write(str(i) + '\t' + str(labels[i]) + '\n')
    print('labels.csv文件创建完成！')
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('G:/python_file/mnist/train', sess.graph)
test_writer = tf.summary.FileWriter('G:/python_file/mnist/test', sess.graph)
# # 创建embedding_var的sprite_image文件（可视化）
# images = mnist.test.images[:show_num] # shape:show_num * 784
# # print(images[0])


def create_sprite_data(images, show_num, single_image_shape):
    n_plots = int(np.ceil(np.sqrt(show_num)))
    sprite_image = np.zeros([n_plots*single_image_shape[0], n_plots*single_image_shape[1]])
    num = 0
    flag = 0
    for i in range(n_plots):
        y1 = i*single_image_shape[1]
        y2 = y1 + single_image_shape[1]
        for j in range(n_plots):
            x1 = j*single_image_shape[0]
            x2 = x1 + single_image_shape[0]
            sprite_image[x1:x2, y1:y2] = 1 - images[num].reshape(single_image_shape)
            num += 1
            # print(num)
            if num > images.shape[0]:
                flag = 1
                break
        if flag:
            break
    cv2.imwrite('G:/python_file/mnist/sprite_image.jpg', sprite_image*255)
    print('sprite_image创建完成！')
# 可视化
saver = tf.train.Saver(tf.global_variables())
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = 'G:/python_file/mnist/projector/labels.tsv'
# embedding.sprite.image_path = 'G:/python_file/mnist/sprite_image.jpg'
# embedding.sprite.single_image_dim.extend(single_image_shape)
projector.visualize_embeddings(tf.summary.FileWriter('G:/python_file/mnist/projector'), config)


init_op = tf.global_variables_initializer()
sess.run(init_op)
for i in range(300):
    batch_xs,batch_ys= mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%50 == 0:
        test = sess.run(merged, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        test_writer.add_summary(test, i)
        train = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        train_writer.add_summary(train, i)

saver.save(sess, 'G:/python_file/mnist/projector/')
