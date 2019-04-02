from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras import activations
from vis.visualization import visualize_activation, visualize_saliency
from vis.utils import utils
import matplotlib.pyplot as plt
from keras import activations
from keras import backend as K
print(K.image_data_format())


# data pre_processing
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
x_tr = x_tr.reshape(-1,28, 28,1)
x_tr = x_tr / 255.0
y_tr = np_utils.to_categorical(y_tr, 10)
x_te = x_te.reshape(-1, 28, 28,1)
x_te = x_te / 255.0
y_te = np_utils.to_categorical(y_te, 10)
# from gene.dataset import *
#
# data = read_data(file='C:/Users/tianping/Desktop/gene_another.csv')

# x, y = preprocess(data, one_hot=False, v2=False)
# print(x.shape)
# print(y.shape)
# X_train, y_train, X_val, y_val, X_test, y_test = split_data(x, y, train_ratio=0.6, val_ratio=0.2)
# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
# print(X_test.shape, y_val.shape)
#
# x_tr = np.concatenate([X_train, X_val])
# y_tr = np.concatenate([y_train, y_val])
# x_te, y_te = X_test, y_test
# print(x_tr.shape)
# print(y_tr.shape)

# create model
model = Sequential()

# conv1
model.add(
    Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu')
)

# pooling1
model.add(
    MaxPool2D(pool_size=(2, 2))
)

# dropout
model.add(Dropout(0.25))

# fc
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, name='preds'))   # change
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, metrics=['accuracy'], loss='categorical_crossentropy')
print(model.layers)
print('training----')
model.fit(x_tr, y_tr, batch_size=32, epochs=5)
print('test')
loss, accuracy = model.evaluate(x_te, y_te)
print('loss' + str(loss))
print('accuracy' + str(accuracy))


# 可视化
# layer_index = utils.find_layer_idx(model, 'preds')
# img = visualize_activation(model, layer_index, input_range=(0.0,1.0), filter_indices=0)
# plt.imshow(img[:,:,0], cmap='jet')
# plt.show()
# visualize_saliency