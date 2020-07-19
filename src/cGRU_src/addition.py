import collections
import numpy as np
import tensorflow as tf
from tensorflow import random_uniform_initializer as urnd_init
from custom_regularizers import complex_dropout
from IPython.core.debugger import Tracer
'''
Other addition functions for our experiments
'''

def matmul_plus_bias_complex(x, num_proj, scope, reuse, bias=True,
                     bias_init=0.0, orthogonal=False):
    """
    Compute Ax + b.
    Arguments:
        x: A real (!) input vector.
        num_proj: The desired dimension of the output.
        scope: This string under which the variables will be
               registered.
        reuse: If this bool is True, the variables will be reused.
        bias: If True a bias will be added.
        bias_init: How to initialize the bias, defaults to zero.
        orthogonal: If true A will be initialized orthogonally
                    and kept orthogonal (make sure to use the
                    Stiefel optimizer if orthogonality is desired).
    Returns:
        Ax + b: A vector of size [batch_size, num_proj]
    """
    in_shape = tf.Tensor.get_shape(x).as_list()
    with tf.variable_scope(scope, reuse=reuse):
        if orthogonal:
            with tf.variable_scope('orthogonal_stiefel', reuse=reuse):
                A = tf.get_variable('gate_O', [in_shape[-1], num_proj],
                                    dtype=tf.complex64,
                                    initializer=tf.orthogonal_initializer())
        else:
            A_real = tf.get_variable('A_real', [in_shape[-1], num_proj], dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())

            A_img = tf.get_variable('A_img', [in_shape[-1], num_proj], dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())
            A = tf.complex(A_real,A_img,name='A')
        if bias:
            b_real = tf.get_variable('bias_real', [num_proj], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_init))

            b_img = tf.get_variable('bias_img', [num_proj], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_init))

            b = tf.complex(b_real,b_img,name='b')
            print('Initializing', tf.contrib.framework.get_name_scope(), 'bias to',
                  bias_init)
            return tf.matmul(x, A) + b
        else:
            return tf.matmul(x, A)

def R_squared(labels, predictions):
    '''
    To calculate R2 for predicted data and actual data
    :param labels: the tensor of actual data
    :param predictions: the tensor of predicted data
    :return: the tensor of data's R2 value
    '''
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels, axis=0)))
    R2 = 1. - tf.div(unexplained_error, total_error)
    return R2