import tensorflow as tf
import numpy as np

#training data
train_1 = np.array( [[1.,2.,3.],
      [3.,4.,5.],
      [8.,5.,7.],
      [7.,1.,8.]] )
train_2 =  np.array( [[1.],
           [0.],
           [0.],
           [1.]] )

"""
3 x [1,3] input , 3 x [1] real answer
hidden layer_1 have 3 input and 2 output
hidden layer_2 have 2 input and 1 output
Activation function : sigmoid
loss function : Mean squared error
Optimizer : Gradient Descent
"""
input_1 = tf.placeholder(tf.float32, shape = [None, 3])
input_2 = tf.placeholder(tf.float32, shape = [None, 1])

weight_1 = tf.get_variable(name='weight_1', shape = [3,2], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1) )
bias_1 = tf.get_variable(name='bias_1', shape = [2], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1) )
layer_1_output =  tf.add(tf.matmul( input_1, weight_1 ), bias_1) 

weight_2 = tf.get_variable(name='weight_2', shape = [2,1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1) )
bias_2 = tf.get_variable(name='bias_2', shape = [1], dtype = tf.float32, initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1) )
layer_2_output = tf.sigmoid( tf.add(tf.matmul( layer_1_output, weight_2 ), bias_2) )

#Mean squared error
#train_2 is our desired output and layer_2_output is output that our network calculated
#Use GradientDescent Minimize the loss to approach our desired output
loss = tf.losses.mean_squared_error(train_2, layer_2_output)
#our goal minimize squared error
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    #initial
    init = tf.global_variables_initializer()
    sess.run( init )
    #train
    for step in range(201) :
        if step % 20 == 0:
            print ('loss : ', sess.run(loss, feed_dict = {input_1: train_1, input_2: train_2})) 
            print ('predict : ', sess.run(layer_2_output, feed_dict = {input_1: train_1}))    
        sess.run(train, feed_dict = {input_1: train_1, input_2: train_2})
