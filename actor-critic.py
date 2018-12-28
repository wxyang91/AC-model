import tensorflow as tf
import numpy as np
class actor_critic:
    def __init__(self, batch_size):
        input_tensor = tf.placeholder(tf.float32,[batch_size, 3])
        Ax = tf.placeholder(tf.float32,[batch_size, 3])
        xQx = tf.placeholder(tf.float32,[batch_size])

        with tf.variable_scope('actor'):
            w1 = tf.get_variable('W1',[3,10],tf.float32,tf.initializers.truncated_normal(stddev=0.6))
            b1 = tf.get_variable('b1',[10],tf.float32,tf.initializers.zeros())
            layer1 = tf.nn.relu(tf.matmul(input,w1) + b1)
            w2 = tf.get_variable('W2', [10, 3], tf.float32, tf.initializers.truncated_normal(stddev=0.6))
            b2 = tf.get_variable('b2', [3], tf.float32, tf.initializers.zeros())
            layer2 = tf.matmul(layer1, w2) + b2
            sigmoid = tf.nn.sigmoid(layer2)
        self.output = sigmoid
        self.input = input_tensor
        def compute_loss(i):
            R = np.mat(np.identity(3) * 1.5)
            B = np.mat([[0],[0],[2]])
            F = tf.constant(B * R * B.T,tf.float32)
            return 2 * tf.reduce_sum(Ax[i] * sigmoid[i]) + xQx[i] - tf.reduce_sum(sigmoid[i] * tf.matmul(F, sigmoid[i]))
        #loss = tf.losses.sigmoid_cross_entropy(y, layer2)
        losses = tf.add_n([tf.square(compute_loss(i)) for i in range(batch_size)]) / tf.constant(batch_size,tf.float32)
        opt = tf.train.AdamOptimizer()
        self.train_opt = opt.minimize(losses)

    def predict(self, x):
        pass
    def observe(self, x):
