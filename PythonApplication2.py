
#Basic
import tensorflow as tf
import numpy as np 
    
    
#set default session
Inter_sess = tf.InteractiveSession()

#Set defalt initializar for all variable in Scope
with tf.variable_scope("defalt_initializar", initializer = tf.constant_initializer(0.1)):
    value = tf.get_variable("value", [1], dtype = tf.float32)
    #use default initializer from variable_scope that we define
    value.initializer.run()
    assert value.eval() == 0.1
    value_redefine = tf.get_variable("value_redefine", [1], dtype = tf.float32, initializer = tf.constant_initializer(0.4) )
    #use default initializer from get_variable that we define
    value_redefine.initializer.run()
    assert value_redefine.eval() == 0.4
    with tf.variable_scope("sub_defalt_initializar"):
        w = tf.get_variable("w", [1], dtype = tf.float32)
        #Inherited default initializer from variable_scope("defalt_initializar") that we define
        w.initializer.run()
        assert w.eval() == 0.1
    
Inter_sess.close()
