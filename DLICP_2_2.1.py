from mpl_toolkits.mplot3d import Axes3D  #required for 3d plotting mandatory
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd


# Improvement to the Linear Regression and new Smoking data set
DATA_FILE = 'Smoking.xls'
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([[sheet.row_values(i)[0], sheet.row_values(i)[1], sheet.row_values(i)[3]] for i in range(1,
                  sheet.nrows)])
n_samples = sheet.nrows - 1

# Initialize placeholders for input X1 status, label X2 age, Y case numbers
X1 = tf.placeholder(tf.float32, name='Status')
X2 = tf.placeholder(tf.float32, name='Age')
Y = tf.placeholder(tf.float32, name='Cases')
# create 2 weights and one bias, initialized to 0
w1 = tf.Variable(0.0, name='weights')
w2 = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')
# Model to predict Y
# Plotting the linear regression of the form by multiplying the two x values (status and age) and adding them
# together

Y_predicted = (X1 * w1 + X2 * w2)/2 + b
# Loss function
loss = tf.square(Y - Y_predicted, name='loss')

# gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('/Users/marcpepperman/Desktop/DeepLearning_ICP_CS4900/tensor_dir', sess.graph)

    # Step 8: train the model
    for i in range(150):  # changed to 20, 50 takes too long
        total_loss = 0
        for x1, x2, y in data:
            # Session runs train_op and fetch values of loss
            _, L = sess.run([optimizer, loss], feed_dict={X1: x1, X2: x2, Y: y})
            total_loss += L

        print('Epoch {0}: {1}'.format(i, total_loss / n_samples))

    # close the writer when you're done using it
    writer.close()

    # weight and balance output
    w1, w2, b = sess.run([w1, w2, b])

# plot the results, The prediction data line is the average of the two weights and variables added together
X1, X2, Y = data.T[0], data.T[1], data.T[2]
plt.plot(X2, Y, 'bo', label='Real data')
plt.plot(X2, (X1 * w1 + X2 * w2)/2 + b, 'r', label='Predicted data')
plt.legend()
plt.show()
