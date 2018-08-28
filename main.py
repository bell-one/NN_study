import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets.cifar10 import load_data


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def get_accuracy(input_x, input_y, session):
    test_accuracy = 0
    for k in range(10):
        batch_x_test, batch_y_test = next_batch(1000, input_x, input_y.eval(session=session))
        test_accuracy = test_accuracy + accuracy.eval(session=session,
                                                      feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0})
    test_accuracy = test_accuracy / 10
    return test_accuracy


# initialize
sess = tf.Session()
(x_train, y_train), (x_test, y_test) = load_data()

# hyper parameters
img_class = 10
width = 32
height = 32
channel = 3
batch_size = 250
training_epochs = 5
learning_rate = 0.0001
keep_drop = 0.75


# neural nets
# input and labels
X = tf.placeholder(tf.float32, shape=[None, width, height, channel])
Y = tf.placeholder(tf.float32, shape=[None, img_class])
keep_prob = tf.placeholder(tf.float32)
y_train = tf.squeeze(tf.one_hot(y_train, img_class), axis=1)
y_test = (tf.squeeze(tf.one_hot(y_test, img_class), axis=1))

# graphs

# 11x11x3 size, 32 layers
W1 = tf.get_variable("W1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([32]))
layer1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
layer1 = tf.nn.relu(layer1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([64]))
layer2 = tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
#layer2 = tf.nn.max_pool(layer2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
layer2 = tf.nn.relu(layer2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]))
layer3 = tf.nn.conv2d(layer2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
layer3 = tf.nn.max_pool(layer3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
layer3 = tf.nn.relu(layer3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
layer4 = tf.nn.conv2d(layer3, W4, strides=[1, 1, 1, 1], padding='SAME') + b4
layer4 = tf.nn.relu(layer4)
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([512]))
layer5 = tf.nn.conv2d(layer4, W5, strides=[1, 1, 1, 1], padding='SAME') + b5
layer5 = tf.nn.relu(layer5)
layer5 = tf.nn.dropout(layer5, keep_prob=keep_prob)


# to fully connected layer, size 8, 8, 128 with 2 pooling
layer5_flat = tf.reshape(layer5, [-1, 16 * 16 * 512])

FC_W1 = tf.get_variable("FC_W1", shape=[16 * 16 * 512, 256], initializer=tf.contrib.layers.xavier_initializer())
FC_b1 = tf.Variable(tf.random_normal([256]))
FC_layer1 = tf.nn.relu(tf.matmul(layer5_flat, FC_W1)+FC_b1)
FC_layer1 = tf.nn.dropout(FC_layer1, keep_prob=keep_prob)

FC_W2 = tf.get_variable("FC_W2", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
FC_b2 = tf.Variable(tf.random_normal([128]))
FC_layer2 = tf.nn.relu(tf.matmul(FC_layer1, FC_W2)+FC_b2)
FC_layer2 = tf.nn.dropout(FC_layer2, keep_prob=keep_prob)

FC_W3 = tf.get_variable("FC_W3", shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())
FC_b3 = tf.Variable(tf.random_normal([10]))
hypo = tf.matmul(FC_layer2, FC_W3)+FC_b3
y_pred = tf.nn.softmax(hypo)

# cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypo, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# test model, accuracy
is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# learn
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(x_train.size / batch_size)
    keep_drop = keep_drop + 0.01 * epoch
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, x_train, y_train.eval(session=sess))
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: keep_drop})
        avg_cost += c / total_batch
        if i % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'total_batch:', '%04d' % (i + 1), 'cost = ', '{:.9f}'.format(c)
                  , "Accuracy: ", get_accuracy(x_test, y_test, sess))
    print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print("Accuracy: ", get_accuracy(x_test, y_test, sess))

saver.save(sess, 'my_test_model')

