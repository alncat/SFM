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

@slim.add_arg_scope
def atrous_deep(net, scope, rates=[2,5,7], depth=256, reuse=None, activation_fn=tf.nn.elu):

    with tf.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rates[0], activation_fn=activation_fn)

        #at_pool3x3_2 = slim.conv2d(at_pool3x3_1, depth, [3, 3], scope="conv_3x3_2", rate=rates[1], activation_fn=activation_fn)

        return at_pool3x3_1

def disp_aspp_u(inputs, args, is_training, reuse, size):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
            upnet1 = slim.conv2d(upnet1, 512, [1, 1])
            upnet1 = slim.conv2d_transpose(upnet1, 512, [3, 3], stride=2)
            aspp_up1 = atrous_deep(upnet1, "ASPP_up1", depth=256, reuse=reuse)
            upnet1 = slim.conv2d(upnet1, 256, [1, 1])

            skip1 = slim.batch_norm(end_points[args.resnet_model + '/block3'], activation_fn=tf.nn.elu)
            skip1 = slim.conv2d(skip1, 256, [1, 1])
            skip1 = slim.conv2d_transpose(skip1, 256, [3, 3], stride=2)

            concat1 = tf.concat([aspp_up1, skip1], axis=3)
            concat1 = slm.conv2d(concat1, 256, [3, 3], rate=4)
            concat1 = concat1 + upnet1
            #upnet2 = slim.conv2d(concat1, 256, [1, 1])
            upnet2 = slim.conv2d_transpose(upnet2, 256, [3, 3], stride=2)

            aspp_up2 = atrous_deep(upnet2, "ASPP_up2", depth=128, reuse=reuse)
            upnet2 = slim.conv2d(upnet2, 128, [1, 1])
            skip2 = slim.batch_norm(end_points[args.resnet_model + '/block2'], activation_fn=tf.nn.elu)
            skip2 = slim.conv2d(skip2, 128, [1, 1])
            skip2 = slim.conv2d_transpose(skip2, 128, [3, 3], stride=2)

            xconcat2 = tf.concat([aspp_up2, skip2], axis=3)
            xconcat2 = slim.conv2d(xconcat2, 128, [3, 3], rate=4)
            xconcat2 = xconcat2 + upnet2
            disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            #upnet3 = slim.conv2d(xconcat2, 128, [1, 1])
            upnet3 = slim.conv2d_transpose(upnet3, 128, [3, 3], stride=2)

            aspp_up3 = atrous_deep(upnet3, "ASPP_up3", [3,7,11], depth=64, reuse=reuse)
            upnet3 = slim.conv2d(upnet3, 64, [1, 1])
            skip3 = end_points[args.resnet_model+'/block1']
            skip3 = slim.batch_norm(skip3, activation_fn=tf.nn.elu)
            skip3 = slim.conv2d(skip3, 64, [1, 1])
            skip3 = slim.conv2d_transpose(skip3, 64, [3, 3], stride=2)
            xconcat3 = tf.concat([aspp_up3, skip3], axis=3)
            xconcat3 = slim.conv2d(xconcat3, 64, [3, 3], rate=6)
            #for output
            xconcat3 = xconcat3 + upnet3
            disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            #upnet4 = slim.conv2d(xconcat3, 64, [1, 1])
            upnet4 = slim.conv2d_transpose(upnet4, 64, [3, 3], stride=2)

            aspp_up4 = atrous_deep(upnet4, "ASPP_up4", [3,7,11], depth=32, reuse=reuse)
            upnet4 = slim.conv2d(upnet4, 32, [1, 1])
            skip4 = end_points[args.resnet_model+'/root_block']
            skip4 = slim.batch_norm(skip4, activation_fn=tf.nn.elu)
            skip4 = slim.conv2d(skip4, 32, [1, 1])
            skip4 = slim.conv2d_transpose(skip4, 32, [3, 3], stride=2)

            xconcat4 = tf.concat([aspp_up4, skip4], axis=3)
            xconcat4 = slim.conv2d(xconcat4, 32, [3, 3], rate=6)
            xconcat4 = xconcat4 + upnet4

            #for output
            disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

def disp_aspp_u_pose(inputs, args, is_training, reuse, size):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu, use_transpose=True)):
        resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            net_size = tf.shape(net)[1:3]
            # resize the preact features
            upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
            upnet1 = slim.conv2d(upnet1, 512, [1, 1])
            upnet1 = slim.conv2d_transpose(upnet1, 512, [3, 3], stride=2)
            aspp_up1 = atrous_deep(upnet1, "ASPP_up1", depth=256, reuse=reuse)
            upnet1 = slim.conv2d(upnet1, 256, [1, 1])

            skip1 = slim.batch_norm(end_points[args.resnet_model + '/block3'], activation_fn=tf.nn.elu)
            skip1 = slim.conv2d(skip1, 128, [1, 1])
            skip1 = slim.conv2d_transpose(skip1, 128, [3, 3], stride=2)

            concat1 = tf.concat([upnet1+aspp_up1, skip1], axis=3)
            upnet2 = slim.conv2d(concat1, 256, [1, 1])
            upnet2 = slim.conv2d_transpose(upnet2, 256, [3, 3], stride=2)

            aspp_up2 = atrous_deep(upnet2, "ASPP_up2", depth=128, reuse=reuse)
            upnet2 = slim.conv2d(upnet2, 128, [1, 1])
            skip2 = slim.batch_norm(end_points[args.resnet_model + '/block2'], activation_fn=tf.nn.elu)
            skip2 = slim.conv2d(skip2, 64, [1, 1])
            skip2 = slim.conv2d_transpose(skip2, 64, [3, 3], stride=2)

            xconcat2 = tf.concat([upnet2+aspp_up2, skip2], axis=3)
            disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            upnet3 = slim.conv2d(xconcat2, 128, [1, 1])
            upnet3 = slim.conv2d_transpose(upnet3, 128, [3, 3], stride=2)

            aspp_up3 = atrous_deep(upnet3, "ASPP_up3", [3,7,11], depth=64, reuse=reuse)
            upnet3 = slim.conv2d(upnet3, 64, [1, 1])
            skip3 = end_points[args.resnet_model+'/block1']
            skip3 = slim.batch_norm(skip3, activation_fn=tf.nn.elu)
            skip3 = slim.conv2d(skip3, 32, [1, 1])
            skip3 = slim.conv2d_transpose(skip3, 32, [3, 3], stride=2)
            xconcat3 = tf.concat([upnet3+aspp_up3, skip3], axis=3)
            #for output
            disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            upnet4 = slim.conv2d(xconcat3, 64, [1, 1])
            upnet4 = slim.conv2d_transpose(upnet4, 64, [3, 3], stride=2)

            aspp_up4 = atrous_deep(upnet4, "ASPP_up4", [3,7,11], depth=32, reuse=reuse)
            upnet4 = slim.conv2d(upnet4, 32, [1, 1])
            #skip4 = end_points[args.resnet_model+'/root_block']
            #skip4 = slim.batch_norm(skip4, activation_fn=tf.nn.elu)
            #skip4 = slim.conv2d(skip4, 16, [1, 1])
            #skip4 = slim.conv2d_transpose(skip4, 16, [3, 3], stride=2)

            xconcat4 = tf.concat([upnet4+aspp_up4], axis=3)

            #for output
            disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

