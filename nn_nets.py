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
                xconcat3 = disp_bottleneck_dd(end_points, args.resnet_model+'/block1', xconcat2, "concat3", args.resnet_model+"/root_block", 128, 64, reuse=reuse, rates=[3,6])
                #for output
                disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
                #upsample4
                xconcat4 = disp_bottleneck_dd(end_points, args.resnet_model+"/root_block", xconcat3, "concat4", args.resnet_model+"/root_block", 64, 32, reuse=reuse, rates=[3,6], use_skip=False)
                #for output
                disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

                return [disp4, disp3, disp2], end_points

def disp_aspp_u_uni(inputs, args, is_training, reuse, size, custom_getter=None):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            root_conv = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='root_conv')
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_slim(inputs,
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

def pose_exp_u_resnet(tgt_image, src_image_stack, args, do_trans=True, is_training=True, reuse=False):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):

        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            root_conv = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='pose_root_conv')

        args.resnet_model = "pose_resnet"
        _, end_points = resnet_v2.resnet_v2_50_slim(inputs,
                                is_training=is_training,
                                global_pool=False,
                                spatial_squeeze=False,
                                output_stride=32,
                                reuse=True,
                                outputs_collections_name="pose_resnet")

        with tf.variable_scope('pose_exp_net', reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                weights_regularizer=slim.l2_regularizer(0.05),
                                activation_fn=tf.nn.elu,
                                outputs_collections=end_points_collection):
                net = slim.batch_norm(end_points[args.resnet_model+'/block4'], activation_fn=tf.nn.elu)
                # Pose specific layers
                with tf.variable_scope('pose', reuse=reuse):
                    cnv6  = resnet_utils.conv2d_same(net, 256, 3, stride=2, scope='cnv6')
                    cnv7  = resnet_utils.conv2d_same(cnv6, 256, 3, stride=2, scope='cnv7')
                    pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred',
                        stride=1, normalizer_fn=None, activation_fn=tf.tanh)
                    pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                    pose_final =  tf.reshape(pose_avg, [-1, num_source, 6])
                    np_scales = np.reshape(np.array([1.,1.,1., np.pi, np.pi, np.pi]), [1,1,6])
                    pose_scales = tf.constant(np_scales, dtype=tf.float32)
                    pose_final *= pose_scales
                # Exp mask specific layers
                if do_trans:
                    with tf.variable_scope('trans', reuse=reuse):
                        upcnv5 = tf.image.resize_bilinear(net, [H//16, W//16])
                        upcnv5 = slim.conv2d(net, 256, [3, 3], stride=1, scope='upcnv5')
                        cnv4 = slim.batch_norm(end_points[args.resnet_model+"/block3"], activation_fn=tf.nn.elu)
                        cnv4 = slim.conv2d(cnv4, 256, [1, 1], stride=1, scope='cnv4')
                        upcnv4 = tf.image.resize_bilinear(tf.concat([upcnv5, cnv4], axis=3), [H//8, W//8])
                        upcnv4 = slim.conv2d(upcnv4, 128, [3, 3], stride=1, scope='upcnv4')

                        cnv3 = slim.batch_norm(end_points[args.resnet_model+"/block2"], activation_fn=tf.nn.elu)
                        cnv3 = slim.conv2d(cnv3, 128, [1, 1], stride=1, scope='cnv3')
                        upcnv3 = tf.image.resize_bilinear(tf.concat([upcnv4, cnv3], axis=3), [H//4, W//4])
                        upcnv3 = slim.conv2d(upcnv3, 128, [3, 3], stride=1, scope='upcnv3')
                        mask3 = 0.2*slim.conv2d(upcnv3, 3*num_source, [3, 3], stride=1, scope='mask3',
                            normalizer_fn=None, activation_fn=tf.tanh)

                        cnv2 = slim.batch_norm(end_points[args.resnet_model+"/block1"], activation_fn=tf.nn.elu)
                        cnv2 = slim.conv2d(cnv2, 64, [1, 1], stride=1, scope='cnv2')
                        upcnv2 = tf.image.resize_bilinear(tf.concat([upcnv3, cnv2], axis=3), [H//2, W//2])
                        upcnv2 = slim.conv2d(upcnv2, 128, [3, 3], stride=1, scope='upcnv2')

                        mask2 = 0.2*slim.conv2d(upcnv2, 3*num_source, [5, 5], stride=1, scope='mask2',
                            normalizer_fn=None, activation_fn=tf.tanh)
                        upcnv1 = tf.image.resize_bilinear(tf.concat([upcnv2, root_conv], axis=3), [H, W])
                        upcnv1 = slim.conv2d(upcnv1, 128, [3, 3], stride=1, scope='upcnv1')
                        mask1 = 0.2*slim.conv2d(upcnv1, 3*num_source, [7, 7], stride=1, scope='mask1',
                            normalizer_fn=None, activation_fn=tf.tanh)
                else:
                    mask1 = None
                    mask2 = None
                    mask3 = None
                end_points = utils.convert_collection_to_dict(end_points_collection)
                return pose_final, [mask1, mask2, mask3], end_points

