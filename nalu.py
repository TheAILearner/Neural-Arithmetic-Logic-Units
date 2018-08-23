def getArithmeticFunction(funcType, X, shape):
    if funcType == "add":
        return (np.sum(X, axis=1, keepdims=True))
    if funcType == "sub":
        return (np.subtract(X[:, 0], X[:, 1])).reshape(shape)
    if funcType == "mul":
        return (np.multiply(X[:, 0], X[:, 1])).reshape(shape)
    if funcType == "div":
        return (np.divide(X[:, 0], X[:, 1])).reshape(shape)


import tensorflow as tf
import numpy as np

X = tf.placeholder(np.float32, shape=[None, 2])
Y = tf.placeholder(np.float32, shape=[None, 1])

#  Defining Neural Arithmetic Logic unit
w_hat = tf.Variable(tf.truncated_normal([2, 1], stddev=0.02))
m_hat = tf.Variable(tf.truncated_normal([2, 1], stddev=0.02))
G = tf.Variable(tf.truncated_normal([2, 1], stddev=0.02))

W = tf.tanh(w_hat) * tf.sigmoid(m_hat)
a = tf.matmul(X, W)
g = tf.sigmoid(tf.matmul(X, G))
m = tf.exp(tf.matmul(tf.log(tf.abs(X) + 1e-7), W))

output = g * a + (1 - g) * m

cost = tf.losses.mean_squared_error(labels=Y, predictions=output)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

init = tf.initialize_all_variables()



functypes = ['add', 'sub', 'mul', 'div']

for functype in functypes:
    with tf.Session() as sess:
        sess.run(init)

        X_train = np.random.randint(1, 100, size=(50000, 2))
        np.random.shuffle(X_train)
        predictions = []
        Y_test = []

        for epoch in range(20):
            batches = 0

            while batches < len(X_train):
                X_training_batches = X_train[batches:batches + 16, :]
                Y_train_batches = getArithmeticFunction(functype, X_training_batches, (16, 1))

                _, c = sess.run([optimizer, cost], feed_dict={X: X_training_batches, Y: Y_train_batches})
                batches += 16
            print("Epoch:", (epoch + 1), "cost =", "{:.15f}".format(c))

        X_test = np.random.randint(1, 1000, size=(1000, 2))
        Y_test = np.round(getArithmeticFunction(functype, X_test, (1000, 1)))

        predictions = np.round(sess.run(output, {X: X_test}))

        count = 0
        for i in range(1000):
            if predictions[i] == Y_test[i]:
                count += 1

        print("accuracy for ", functype, "  :", count / 1000)
        del X_train, X_test
    sess.close()
