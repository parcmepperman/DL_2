from mpl_toolkits.mplot3d import Axes3D  #required for 3d plotting mandatory
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd


DATA_FILE = 'USA_Housing.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([[sheet.row_values(i)[0], sheet.row_values(i)[5]] for i in range(1,
                  sheet.nrows)])
n_samples = sheet.nrows - 1

# Initialize placeholders for input X1 income and Y
X1 = tf.placeholder(tf.float32, name='Income')

Y = tf.placeholder(tf.float32, name='Price')
# create 2 weights and one bias, initialized to 0
w1 = tf.Variable(0.0, name='weights')

b = tf.Variable(0.0, name='bias')
# Model to predict Y
# Plotting the linear regression of the form by multiplying the two x values (status and age) and adding them
# together

Y_predicted = X1 * w1 + b
# Loss function
loss = tf.square(Y - Y_predicted, name='loss')
                                                                                    # changed to optimize data to
                                                                                    # 0.1 * 10^-12
# gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000000000001).minimize(loss)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/Users/marcpepperman/Desktop/DeepLearning_ICP_CS4900/tensor_dir', sess.graph)

    for i in range(20):                                                          # changed to 20, 50 takes too long
        total_loss = 0
        for x1, y in data:
                                                                     # Session runs train_op and fetch values of loss
            _, L = sess.run([optimizer, loss], feed_dict={X1: x1, Y: y})
            total_loss += L

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    writer.close()

    w1, b = sess.run([w1, b])


# plot the results
X1, Y = data.T[0], data.T[1]
plt.plot(X1, Y, 'bo', label='Real data')
plt.plot(X1, w1 * X1 + b, 'r', label='Predicted data')
plt.legend()
plt.show()
