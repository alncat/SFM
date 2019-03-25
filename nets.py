from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
import cspn
from resnet import resnet_v2, resnet_utils
slim = tf.contrib.slim

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_trans_net(tgt_image, src_image_stack, do_trans=True, is_training=True, reuse=False):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose', reuse=reuse):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred',
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_trans:
                with tf.variable_scope('trans', reuse=reuse):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = 0.001*slim.conv2d(upcnv3, num_source, [3, 3], stride=1, scope='mask3',
                        normalizer_fn=None, activation_fn=None)

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = 0.001*slim.conv2d(upcnv2, num_source, [5, 5], stride=1, scope='mask2',
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = 0.001*slim.conv2d(upcnv1, num_source, [7, 7], stride=1, scope='mask1',
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3], end_points

def pose_trans_u_net(tgt_image, src_image_stack, do_trans=True, is_training=True, reuse=False):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose', reuse=reuse):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred',
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_trans:
                with tf.variable_scope('trans', reuse=reuse):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = 0.2*slim.conv2d(upcnv3, 3*num_source, [3, 3], stride=1, scope='mask3',
                        normalizer_fn=None, activation_fn=tf.tanh)

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = 0.2*slim.conv2d(upcnv2, 3*num_source, [5, 5], stride=1, scope='mask2',
                        normalizer_fn=None, activation_fn=tf.tanh)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = 0.2*slim.conv2d(upcnv1, 3*num_source, [7, 7], stride=1, scope='mask1',
                        normalizer_fn=None, activation_fn=tf.tanh)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3], end_points

def pose_exp_u_net(tgt_image, src_image_stack, do_trans=True, is_training=True, reuse=False):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose', reuse=reuse):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred',
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_trans:
                with tf.variable_scope('trans', reuse=reuse):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(tf.concat([upcnv5, cnv4], axis=3), 128, [3, 3], stride=2, scope='upcnv4')

                    upcnv3 = slim.conv2d_transpose(tf.concat([upcnv4, cnv3], axis=3), 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = 0.2*slim.conv2d(upcnv3, 3*num_source, [3, 3], stride=1, scope='mask3',
                        normalizer_fn=None, activation_fn=tf.tanh)

                    upcnv2 = slim.conv2d_transpose(tf.concat([upcnv3,  cnv2], axis=3), 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = 0.2*slim.conv2d(upcnv2, 3*num_source, [5, 5], stride=1, scope='mask2',
                        normalizer_fn=None, activation_fn=tf.tanh)

                    upcnv1 = slim.conv2d_transpose(tf.concat([upcnv2, cnv1], axis=3), 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = 0.2*slim.conv2d(upcnv1, 3*num_source, [7, 7], stride=1, scope='mask1',
                        normalizer_fn=None, activation_fn=tf.tanh)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3], end_points

def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True, reuse=False):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose', reuse=reuse):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred',
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp', reuse=reuse):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4',
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3',
                        normalizer_fn=None, activation_fn=None)

                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2',
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1',
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points

def disp_net(tgt_image, is_training=True, reuse=False):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], end_points

def disp_net_cspn(tgt_image, is_training=True, reuse=False):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net', reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.01),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            #disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1,
            #    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            #disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            #disp4 = 1./disp4
            #cspn_in4 = slim.conv2d(icnv4, 8, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, scope='affinity4')
            #disp4 = cspn.cspn(cspn_in4, disp4, 1)

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])
            disp3 = 1./disp3
            cspn_in3 = slim.conv2d(icnv3, 8, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, scope='affinity3')
            disp3 = cspn.cspn(cspn_in3, disp3, 1)

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            disp2 = 1./disp2
            cspn_in2 = slim.conv2d(icnv2, 8, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, scope='affinity2')
            disp2 = cspn.cspn(cspn_in2, disp2, 2)

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
            disp1 = 1./disp1
            cspn_in = slim.conv2d(icnv1, 8, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, scope='affinity')
            disp1 = cspn.cspn(cspn_in, disp1, 4)

            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3], end_points

def disp_bottleneck(end_points, concat, scope, skip_block, in_channels, out_channels, reuse=False, rates=[2,4]):
    with tf.variable_scope(scope, reuse=reuse):
        upnet = slim.conv2d_transpose(concat, in_channels, [3, 3], stride=2)
        aspp_up = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[0])
        upnet = slim.conv2d(upnet, out_channels, [1, 1], normalizer_fn=None, activation_fn=None)
        skip = slim.batch_norm(end_points[skip_block], activation_fn=tf.nn.elu)
        skip = slim.conv2d(skip, out_channels, [1, 1], normalizer_fn=None, activation_fn=None)
        skip = slim.conv2d_transpose(skip, out_channels, [3, 3], stride=2)

        xconcat = tf.concat([aspp_up, skip], axis=3)
        xconcat = slim.conv2d(xconcat, out_channels, [3, 3], rate=rates[1], normalizer_fn=None, activation_fn=None)
        xconcat = xconcat + upnet
        xconcat = slim.batch_norm(xconcat, activation_fn=tf.nn.elu)
        return xconcat

def disp_bottleneck_nt(end_points, concat, scope, skip_block, in_channels, out_channels, reuse=False, rates=[2,4], use_skip=True):
    with tf.variable_scope(scope, reuse=reuse):
        iH, iW = concat.get_shape()[1], concat.get_shape()[2]
        upnet = tf.image.resize_bilinear(concat, [iH*2, iW*2])
        upnet = slim.conv2d(upnet, in_channels, [1, 1], activation_fn=None)
        aspp_up = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[0])
        #upnet = slim.conv2d(upnet, out_channels, [1, 1], activation_fn=None)
        upnet = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[1])
        if use_skip:
            skip = slim.batch_norm(end_points[skip_block], activation_fn=tf.nn.elu)
            skip = slim.conv2d(skip, out_channels, [1, 1])
        else:
            skip = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[1]*2)
        xconcat = tf.concat([aspp_up, upnet, skip], axis=3)
        xconcat = slim.conv2d(xconcat, out_channels, [1, 1], activation_fn=None)
        xd = slim.conv2d(xconcat, out_channels, [3, 3])
        xd = tf.concat([xconcat, xd], axis=3)
        xd = slim.conv2d(xd, out_channels, [1, 1], activation_fn=None)
        #xconcat = tf.concat([aspp_up, skip], axis=3)
        #xconcat = slim.conv2d(xconcat, out_channels, [3, 3], rate=rates[1], activation_fn=None)
        #xconcat = tf.nn.elu(xconcat + upnet)
        return xd

def disp_bottleneck_d(end_points, concat, scope, skip_block, in_channels, out_channels, reuse=False, rates=[2,4], use_skip=True):
    with tf.variable_scope(scope, reuse=reuse):
        iH, iW = concat.get_shape()[1], concat.get_shape()[2]
        upnet = tf.image.resize_bilinear(concat, [iH*2, iW*2])
        upnet = slim.conv2d(upnet, in_channels, [1, 1], activation_fn=None)
        supnet = slim.conv2d(upnet, out_channels, [1, 1])
        aspp_up = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[0])
        aspp_up2 = slim.conv2d(tf.concat([supnet, aspp_up], axis=3), out_channels, [3, 3], rate=rates[1])
        if use_skip:
            skip = slim.batch_norm(end_points[skip_block], activation_fn=tf.nn.elu)
            skip = slim.conv2d(skip, out_channels, [1, 1])
        else:
            skip = slim.conv2d(tf.concat([aspp_up, aspp_up2], axis=3), out_channels, [3, 3], rate=rates[1]*2)
        xconcat = tf.concat([supnet, aspp_up, aspp_up2, skip], axis=3)
        xconcat = slim.conv2d(xconcat, out_channels, [1, 1], activation_fn=None)
        return xconcat

def disp_bottleneck_dense(end_points, concat, scope, skip_block, in_channels, out_channels, reuse=False, rates=[2,4], use_skip=True, use_skip2=True, skip2=None):
    with tf.variable_scope(scope, reuse=reuse):
        iH, iW = concat.get_shape()[1], concat.get_shape()[2]
        upnet = tf.image.resize_bilinear(concat, [iH*2, iW*2])
        upnet = slim.conv2d(upnet, in_channels, [1, 1], activation_fn=None)
        supnet = slim.conv2d(upnet, out_channels, [1, 1])
        aspp_up = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[0])
        aspp_up2 = slim.conv2d(tf.concat([supnet, aspp_up], axis=3), out_channels, [3, 3], rate=rates[1])
        if use_skip:
            skip = slim.batch_norm(end_points[skip_block], activation_fn=tf.nn.elu)
            skip = slim.conv2d(skip, out_channels, [1, 1])
        else:
            skip = slim.conv2d(tf.concat([aspp_up, aspp_up2], axis=3), out_channels, [3, 3], rate=rates[1]*2)
        if use_skip2:
            skip2 = tf.image.resize_bilinear(skip2, [iH*2, iW*2])
            skip2 = slim.conv2d(skip2, out_channels, [1, 1])
            xconcat = tf.concat([supnet, aspp_up, aspp_up2, skip, skip2], axis=3)
        else:
            xconcat = tf.concat([supnet, aspp_up, aspp_up2, skip], axis=3)
        xconcat = slim.conv2d(xconcat, out_channels, [1, 1], activation_fn=None)
        return xconcat, upnet

def disp_bottleneck_dd(end_points, skip2, concat, scope, skip_block, in_channels, out_channels, reuse=False, rates=[2,4], use_skip=True, use_skip2=True):
    with tf.variable_scope(scope, reuse=reuse):
        iH, iW = concat.get_shape()[1], concat.get_shape()[2]
        if use_skip2:
            skip2 = slim.batch_norm(end_points[skip2], activation_fn=tf.nn.elu)
            skip2 = slim.conv2d(skip2, out_channels, [1, 1])
            skip2 = tf.image.resize_bilinear(skip2, [iH*2, iW*2])
        upnet = tf.image.resize_bilinear(concat, [iH*2, iW*2])
        if use_skip2:
            upnet = slim.conv2d(upnet, in_channels, [1, 1], activation_fn=None)
        supnet = slim.conv2d(upnet, out_channels, [1, 1])
        aspp_up = slim.conv2d(upnet, out_channels, [3, 3], rate=rates[0])
        aspp_up2 = slim.conv2d(tf.concat([supnet, aspp_up], axis=3), out_channels, [3, 3], rate=rates[1])
        if use_skip:
            skip = slim.batch_norm(end_points[skip_block], activation_fn=tf.nn.elu)
            skip = slim.conv2d(skip, out_channels, [1, 1])
        else:
            skip = slim.conv2d(tf.concat([aspp_up, aspp_up2], axis=3), out_channels, [3, 3], rate=rates[1]*2)
        xconcat = tf.concat([supnet, aspp_up, aspp_up2, skip], axis=3)
        if use_skip2:
            xconcat = tf.concat([xconcat, skip2], axis=3)
        xconcat = slim.conv2d(xconcat, out_channels, [1, 1], activation_fn=None)
        return xconcat

def disp_aspp_u(inputs, args, is_training, reuse, size, custom_getter=None):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_mod(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=32,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse, custom_getter=custom_getter):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            upnet = slim.batch_norm(net, activation_fn=tf.nn.elu)
            upnet = slim.conv2d(upnet, 512, [1, 1], normalizer_fn=None, activation_fn=None)
            upnet = slim.conv2d_transpose(upnet, 512, [3, 3], stride=2)
            aspp_up = slim.conv2d(upnet, 512, [3, 3], rate=2)
            #upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
            upnet1 = slim.conv2d(tf.concat([upnet, aspp_up], axis=3), 512, [1, 1], activation_fn=None)
            concat1 = disp_bottleneck(end_points, upnet1, "concat1", args.resnet_model+"/block3", 512, 256, reuse=reuse)

            xconcat2 = disp_bottleneck(end_points, concat1, "concat2", args.resnet_model+"/block2", 256, 128, reuse=reuse)
            disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            #upnet3 = slim.conv2d(xconcat2, 128, [1, 1])
            xconcat3 = disp_bottleneck(end_points, xconcat2, "concat3", args.resnet_model+"/block1", 128, 64, reuse=reuse, rates=[3,6])
            #for output
            disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            xconcat4 = disp_bottleneck(end_points, xconcat3, "concat4", args.resnet_model+"/root_block", 64, 32, reuse=reuse, rates=[3,6])

            #for output
            disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

def disp_aspp_u_nt(inputs, args, is_training, reuse, size, custom_getter=None):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_mod(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=32,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse, custom_getter=custom_getter):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose]):
                upnet = slim.batch_norm(net, activation_fn=tf.nn.elu)
                upnet = slim.conv2d(upnet, 1024, [1, 1], activation_fn=None)
                #upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
                concat = disp_bottleneck_nt(end_points, upnet, "concat", args.resnet_model+"/block3", 1024, 512, reuse=reuse)
                concat1 = disp_bottleneck_nt(end_points, concat, "concat1", args.resnet_model+"/block2", 512, 256, reuse=reuse)
                xconcat2 = disp_bottleneck_nt(end_points, concat1, "concat2", args.resnet_model+"/block1", 256, 128, reuse=reuse)
                disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

                #upsampe3
                xconcat3 = disp_bottleneck_nt(end_points, xconcat2, "concat3", args.resnet_model+"/root_block", 128, 64, reuse=reuse, rates=[3,6])
                #for output
                disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

                #upsample4
                xconcat4 = disp_bottleneck_nt(end_points, xconcat3, "concat4", args.resnet_model+"/root_block", 64, 32, reuse=reuse, rates=[3,6], use_skip=False)

                #for output
                disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

                return [disp4, disp3, disp2], end_points

def disp_aspp_u_d(inputs, args, is_training, reuse, size, custom_getter=None):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_mod(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=32,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse, custom_getter=custom_getter):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose]):
                upnet = slim.batch_norm(net, activation_fn=tf.nn.elu)
                upnet = slim.conv2d(upnet, 1024, [1, 1], activation_fn=None)
                #upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
                concat = disp_bottleneck_d(end_points, upnet, "concat", args.resnet_model+"/block3", 1024, 512, reuse=reuse)
                concat1 = disp_bottleneck_d(end_points, concat, "concat1", args.resnet_model+"/block2", 512, 256, reuse=reuse)
                xconcat2 = disp_bottleneck_d(end_points, concat1, "concat2", args.resnet_model+"/block1", 256, 128, reuse=reuse)
                disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

                #upsampe3
                xconcat3 = disp_bottleneck_d(end_points, xconcat2, "concat3", args.resnet_model+"/root_block", 128, 64, reuse=reuse, rates=[3,6])
                #for output
                disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

                #upsample4
                xconcat4 = disp_bottleneck_d(end_points, xconcat3, "concat4", args.resnet_model+"/root_block", 64, 32, reuse=reuse, rates=[3,6], use_skip=False)

                #for output
                disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

                return [disp4, disp3, disp2], end_points

def disp_aspp_u_dense(inputs, args, is_training, reuse, size, custom_getter=None):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_mod(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=32,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse, custom_getter=custom_getter):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose]):
                upnet = slim.batch_norm(net, activation_fn=tf.nn.elu)
                upnet = slim.conv2d(upnet, 1024, [1, 1], activation_fn=None)
                #upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
                concat, _ = disp_bottleneck_dense(end_points, upnet, "concat", args.resnet_model+"/block3", 1024, 512, reuse=reuse, use_skip2=False)
                concat1, skip1 = disp_bottleneck_dense(end_points, concat, "concat1", args.resnet_model+"/block2", 512, 256, reuse=reuse, use_skip2=False)
                xconcat2, skip2 = disp_bottleneck_dense(end_points, concat1, "concat2", args.resnet_model+"/block1", 256, 128, reuse=reuse, skip2=skip1)
                disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

                #upsampe3
                xconcat3, skip3 = disp_bottleneck_dense(end_points, xconcat2, "concat3", args.resnet_model+"/root_block", 128, 64, reuse=reuse, rates=[3,6], skip2=skip2)
                #for output
                disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

                #upsample4
                xconcat4, _ = disp_bottleneck_dense(end_points, xconcat3, "concat4", args.resnet_model+"/root_block", 64, 32, reuse=reuse, rates=[3,6], use_skip=False, skip2=skip3)

                #for output
                disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

                return [disp4, disp3, disp2], end_points

def disp_aspp_u_dd(inputs, args, is_training, reuse, size, custom_getter=None):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_mod(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=32,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse, custom_getter=custom_getter):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose]):
                upnet = slim.batch_norm(net, activation_fn=tf.nn.elu)
                upnet = slim.conv2d(upnet, 1024, [1, 1], activation_fn=None)
                #upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
                concat = disp_bottleneck_dd(end_points, None, upnet, "concat", args.resnet_model+"/block3", 1024, 512, reuse=reuse, use_skip2=False)
                concat1 = disp_bottleneck_dd(end_points, args.resnet_model+'/block3', concat, "concat1", args.resnet_model+"/block2", 512, 256, reuse=reuse, use_skip2=True)
                xconcat2 = disp_bottleneck_dd(end_points, args.resnet_model+'/block2', concat1, "concat2", args.resnet_model+"/block1", 256, 128, reuse=reuse)
                disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
                #upsampe3
                xconcat3 = disp_bottleneck_dd(end_points, args.resnet_model+'/block2', xconcat2, "concat3", args.resnet_model+"/root_block", 128, 64, reuse=reuse, rates=[3,6])
                #for output
                disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
                #upsample4
                xconcat4 = disp_bottleneck_dd(end_points, args.resnet_model+"/root_block", xconcat3, "concat4", args.resnet_model+"/root_block", 64, 32, reuse=reuse, rates=[3,6], use_skip=False)
                #for output
                disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

                return [disp4, disp3, disp2], end_points

