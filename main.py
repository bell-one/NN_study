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


# initialize
sess = tf.Session()
(x_train, y_train), (x_test, y_test) = load_data()

# hyper parameters
img_class = 10
width = 32
height = 32
batch_size = 250
training_epochs = 15
learning_rate = 0.001

## neural nets
# input and labels
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
y_train = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

# graphs

# 3x3x1 size, 64 layers
W1 = tf.get_variable("W1", shape=[7, 7, 3, 32],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([32]))
layer1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
layer1 = tf.nn.relu(layer1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[5, 5, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([64]))
layer2 = tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
layer2 = tf.nn.relu(layer2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[3, 3, 64, 128],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]))
layer3 = tf.nn.conv2d(layer2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
layer3 = tf.nn.relu(layer3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[3, 3, 128, 128],initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([128]))
layer4 = tf.nn.conv2d(layer3, W4, strides=[1, 1, 1, 1], padding='SAME') + b4
layer4 = tf.nn.relu(layer4)
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[3, 3, 128, 128],initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([128]))
layer5 = tf.nn.conv2d(layer4, W5, strides=[1, 1, 1, 1], padding='SAME') + b5
layer5 = tf.nn.relu(layer5)
layer5 = tf.nn.dropout(layer5, keep_prob=keep_prob)

# to fully connected layer, size 32, 32, 128 without pooling
layer5_flat = tf.reshape(layer5, [-1, 32 * 32 * 128])

FC_W1 = tf.get_variable("FC_W1", shape=[32 * 32 * 128, 128],initializer=tf.contrib.layers.xavier_initializer())
FC_b1 = tf.Variable(tf.random_normal([128]))
FC_layer1 = tf.nn.relu(tf.matmul(layer5_flat, FC_W1)+FC_b1)
FC_layer1 = tf.nn.dropout(FC_layer1, keep_prob=keep_prob)

FC_W2 = tf.get_variable("FC_W2", shape=[128, 10],initializer=tf.contrib.layers.xavier_initializer())
FC_b2 = tf.Variable(tf.random_normal([10]))
hypo = tf.matmul(FC_layer1, FC_W2)+FC_b2
y_pred = tf.nn.softmax(hypo)

# cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypo, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# test model, accuracy
is_correct = tf.equal(tf.argmax(hypo, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# learn
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(x_train.size / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, x_train, y_train.eval(session=sess))
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.75})
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: x_test, Y: y_test, keep_prob: 1}))