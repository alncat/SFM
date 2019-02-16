from __future__ import division
import tensorflow as tf
import numpy as np

def cspn(guidance, blur_depth, iterations = 8):
    gates = []
    for i in range(8):
        gates.append(tf.slice(guidance, [0,0,0,i], [-1,-1,-1,1]))

    spn_kernel = 3

    result_depth = tf.identity(blur_depth)
    for i in range(iterations):
        elewise_max_gates = []
        for j in range(len(gates)):
            elewise_max_gates.append(eight_way_propagation(gates[j], result_depth, spn_kernel, j))
        #elewise_max_gates = tf.concat(elewise_max_gates, axis = 3)
        #result_depths = tf.nn.top_k(elewise_max_gates, 5).values
        #result_depth = tf.reduce_mean(elewise_max_gates, axis = 3, keepdims=True)
        #result_depth = 0.5*(tf.slice(result_depths, [0,0,0,4], [-1,-1,-1,1]) + tf.slice(result_depths, [0,0,0,3], [-1,-1,-1,1]))
        result_depth = max_of_8_tensor(elewise_max_gates)
    return result_depth



def eight_way_propagation(weight_matrix, blur_matrix, kernel, i):
    with tf.name_scope('eight_way') as scope:
        value = np.ones([kernel,kernel,1,1])
        value[(kernel-1)//2, (kernel-1)//2, 0, 0] = 0
        #filter_ = tf.constant(np.reshape([[1.,1.,1.],[1.,0.,1.],[1.,1.,1.]], [3, 3, 1, 1]),shape=[kernel,kernel,1,1], name='weights')
        filter_ = tf.constant(value, name='weights', dtype=tf.float32)
        abs_weight = tf.abs(weight_matrix)
        abs_weight_sum = tf.nn.conv2d(abs_weight, filter_, [1, 1, 1, 1], padding='SAME')
        weight_sum = tf.nn.conv2d(weight_matrix, filter_, [1, 1, 1, 1], padding='SAME')
        others = tf.nn.conv2d(tf.multiply(weight_matrix, blur_matrix), filter_, [1, 1, 1, 1], padding='SAME')
        out = tf.divide(tf.multiply(abs_weight_sum - weight_sum, blur_matrix) + others, abs_weight_sum)

    return out

def max_of_4_tensor(elements):
    max_element1 = tf.maximum(elements[0], elements[1])
    max_element2 = tf.maximum(elements[2], elements[3])
    return tf.maximum(max_element1, max_element2)

def max_of_8_tensor(elements):
    max_element1 = max_of_4_tensor(elements[0:4])
    max_element2 = max_of_4_tensor(elements[4:])
    return tf.minimum(max_element1, max_element2)
