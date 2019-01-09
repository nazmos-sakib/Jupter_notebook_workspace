#!/usr/bin/env python
import tensorflow as tf

scalar = tf.constant([2])
vector = tf.constant([3, 4, 5])
matrix = tf.constant([[6, 7, 8], [9, 10, 11], [12, 13, 14]])

tensor = tf.constant([ [[6, 7, 8], [9, 10, 11], [12, 13, 14]], [[15, 16, 17],[18,19,20],[21,22,23]],[[15, 16, 17],[18,19,20],[21,22,23]]])

with tf.Session() as session:
    result = session.run(scalar)
    print ("scalar data example:\n",result)
    
    result = session.run(vector)
    print("\nVector Data:\n", result)

    result = session.run(matrix)
    print("\nMatrix:\n", result)
    
    result = session.run(tensor)
    print("\nTensor:\n", result)
