import tensorflow as tf
import utils
slim = tf.contrib.slim
from resnet import resnet_v2, resnet_utils

# ImageNet mean statistics
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, depth=256, reuse=None, activation_fn=tf.nn.elu):

    with tf.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                           activation_fn=activation_fn)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=activation_fn)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=1, activation_fn=activation_fn)

        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=3, activation_fn=activation_fn)

        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=5, activation_fn=activation_fn)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=activation_fn)
        return net

#257
@slim.add_arg_scope
def atrous_deep(net, scope, rates=[2,5,7], depth=256, reuse=None, activation_fn=tf.nn.elu):

    with tf.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rates[0], activation_fn=activation_fn)

        at_pool3x3_2 = slim.conv2d(at_pool3x3_1, depth, [3, 3], scope="conv_3x3_2", rate=rates[1], activation_fn=activation_fn)
        #added a lower level feature
        #at_pool3x3_2 = tf.concat([at_pool3x3_2, at_pool3x3_1], axis=3)

        #at_pool3x3_3 = slim.conv2d(at_pool3x3_2, depth, [3, 3], scope="conv_3x3_3", rate=rates[2], activation_fn=None, normalizer_fn=None)
        #at_pool3x3_3 = slim.conv2d(at_pool3x3_2, depth, [3, 3], scope="conv_3x3_3", rate=rates[2])

        return at_pool3x3_2

@slim.add_arg_scope
def bilinear_deep(net, scope, in_depth=256, rates=[2,3,5], depth=256, reuse=None, activation_fn=tf.nn.elu):

    with tf.variable_scope(scope, reuse=reuse):

        at_pool3x3_1 = slim.batch_norm(utils.bilinear_conv2d(net, "conv_3x3_1", 3, in_depth, depth, rate=rates[0], reuse=reuse, activation_fn=activation_fn))

        at_pool3x3_2 = slim.batch_norm(utils.bilinear_conv2d(at_pool3x3_1, "conv_3x3_2", 3, depth, depth, rate=rates[1], reuse=reuse, activation_fn=activation_fn))

        at_pool3x3_3 = slim.batch_norm(utils.bilinear_conv2d(at_pool3x3_2, "conv_3x3_3", 3, depth, depth, rate=rates[2], reuse=reuse, activation_fn=activation_fn))

        return at_pool3x3_3

def disp_new(inputs, args, is_training, reuse):

    # mean subtraction normalization
    inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]
    out_depth = 256

    # inputs has shape - Original: [batch, 513, 513, 3]
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon)):
        resnet = getattr(resnet_v2, args.resnet_model)
        _, end_points = resnet(inputs,
                               #args.number_of_classes,
                               256,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):

            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net = slim.conv2d(net, out_depth, [1, 1], activation_fn=tf.nn.relu,
                              scope='logits')

            size = tf.shape(inputs)[1:3]
            net_size = tf.shape(net)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)

            upnet1 = tf.image.resize_bilinear(net, [net_size[0]*2, net_size[1]*2])
            sccnv1 = slim.conv2d(upnet1, 128, [3, 3], activation_fn=None, stride=1, scope='sccnv1')

            upcnv1 = slim.conv2d_transpose(net, 128,  [3, 3], stride=2, scope='upcnv1')
            #i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = sccnv1 + slim.conv2d(upcnv1, 128,  [3, 3], activation_fn=None, stride=1, scope='icnv1')

            upicnv1 = tf.image.resize_bilinear(net, [net_size[0]*4, net_size[1]*4])
            sccnv2 = slim.conv2d(upicnv1, 64, [3, 3], activation_fn=None, stride=1, scope='sccnv2')

            upcnv2 = slim.conv2d_transpose(icnv1, 64, [3, 3], stride=2, scope='upcnv2')
            icnv2 = sccnv2 + slim.conv2d(upcnv2, 64, [3, 3], stride=1, scope='icnv2')

            disp2 = DISP_SCALING*slim.conv2d(icnv2, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            upicnv2 = tf.image.resize_bilinear(net, [net_size[0]*8, net_size[1]*8])

            sccnv3 = slim.conv2d(upicnv2, 32, [3, 3], activation_fn=None, stride=1, scope='sccnv3')

            upcnv3 = slim.conv2d_transpose(icnv2, 32, [3, 3], stride=2, scope='upcnv3')
            icnv3 = sccnv3 + slim.conv2d(upcnv3, 32, [3, 3], stride=1, scope='icnv3')
            disp3 = DISP_SCALING*slim.conv2d(icnv3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            upicnv3 = tf.image.resize_bilinear(net, [net_size[0]*16, net_size[1]*16])

            sccnv4 = slim.conv2d(upicnv3, 16, [3, 3], activation_fn=None, stride=1, scope='sccnv4')

            upcnv4 = slim.conv2d_transpose(icnv3, 16, [3, 3], stride=2, scope='upcnv4')
            icnv4 = sccnv4 + slim.conv2d(upcnv4, 16, [3, 3], stride=1, scope='icnv4')

            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

def disp_bilinear(inputs, args, is_training, reuse, size):

    out_depth = 256

    # inputs has shape - Original: [batch, 513, 513, 3]
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu)):
        resnet = getattr(resnet_v2, args.resnet_model)
        _, end_points = resnet(inputs,
                               #args.number_of_classes,
                               256,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):

            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            #size = tf.shape(inputs)[1:3]
            net_size = tf.shape(net)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)

            #a lock for up sampling
            upnet1 = tf.image.resize_bilinear(net, [size[0]//8, size[1]//8])
            aspp_up1 = bilinear_deep(upnet1, "ASPP_up1", in_depth=256, depth=128, reuse=reuse)
            concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')

            #upsample2
            upicnv1 = tf.image.resize_bilinear(icnv1, [size[0]//4, size[1]//4])

            aspp_up2 = bilinear_deep(upicnv1, "ASPP_up2", in_depth=128, depth=64, reuse=reuse)
            concat2 = tf.concat([upicnv1, aspp_up2], axis=3)

            #for output
            disp2 = DISP_SCALING*slim.conv2d(concat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            upicnv2 = tf.image.resize_bilinear(aspp_up2, [size[0]//2, size[1]//2])

            aspp_up3 = bilinear_deep(upicnv2, "ASPP_up3", rates = [3,5,7], in_depth=64, depth=32, reuse=reuse)
            concat3 = tf.concat([upicnv2, aspp_up3], axis=3)

            #for output
            disp3 = DISP_SCALING*slim.conv2d(concat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            upicnv3 = tf.image.resize_bilinear(aspp_up3, [size[0], size[1]])

            aspp_up4 = bilinear_deep(upicnv3, "ASPP_up4", rates = [3,7,11], in_depth=32, depth=16, reuse=reuse)
            concat4 = tf.concat([upicnv3, aspp_up4], axis=3)

            #for output
            disp4  = DISP_SCALING * slim.conv2d(concat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

def disp_aspp(inputs, args, is_training, reuse, size):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu)):
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

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net_size = tf.shape(net)[1:3]
            # resize the output logits to match the labels dimensions
            upnet1 = tf.image.resize_bilinear(net, [size[0]//8, size[1]//8])
            aspp_up1 = atrous_deep(upnet1, "ASPP_up1", depth=128, reuse=reuse)
            #concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            #upnet1 = slim.conv2d(upnet1, 128, [1, 1])
            #concat1 = slim.batch_norm(tf.multiply(upnet1, aspp_up1))
            concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            #icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')
            #block3 = slim.conv2d(end_points[args.resnet_model + '/block3'], 128, [1, 1])
            #icnv1 = tf.concat([concat1, block3], axis=3)
            icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')

            #upsample2
            upicnv1 = tf.image.resize_bilinear(icnv1, [size[0]//4, size[1]//4])

            aspp_up2 = atrous_deep(upicnv1, "ASPP_up2", [3, 5, 7], depth=64, reuse=reuse)
            #concat2 = tf.concat([upicnv1, aspp_up2], axis=3)
            #upicnv1 = slim.conv2d(upicnv1, 64, [1, 1])
            skip2 = slim.batch_norm(end_points[args.resnet_model + '/block2'], activation_fn=tf.nn.elu)
            skip2 = slim.conv2d(skip2, 64, [1, 1])
            skip2 = tf.image.resize_bilinear(skip2, [size[0]//4, size[1]//4])

            #xconcat2 = slim.batch_norm(tf.multiply(upicnv1, aspp_up2))
            xconcat2 = tf.concat([upicnv1, aspp_up2, skip2], axis=3)
            #block2 = slim.conv2d(end_points[args.resnet_model + '/block2'], 64, [1,1])
            #xconcat2 = tf.concat([xconcat2, block2], axis=3)

            #for output
            disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            xconcat2 = slim.conv2d(xconcat2, 64, [1, 1], stride=1, scope='xconcat2')
            upicnv2 = tf.image.resize_bilinear(xconcat2, [size[0]//2, size[1]//2])

            aspp_up3 = atrous_deep(upicnv2, "ASPP_up3", [3,7,11], depth=32, reuse=reuse)
            #concat3 = tf.concat([upicnv2, aspp_up3], axis=3)
            #upicnv2 = slim.conv2d(upicnv2, 32, [1, 1])
            #xconcat3 = slim.batch_norm(tf.multiply(upicnv2, aspp_up3))
            xconcat3 = tf.concat([upicnv2, aspp_up3], axis=3)
            #block1 = slim.conv2d(end_points[args.resnet_model + '/block1'], 32, [1, 1])
            #xconcat3 = tf.concat([xconcat3, block1], axis=3)

            #for output
            disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            xconcat3 = slim.conv2d(xconcat3, 32, [1, 1], stride=1, scope='xconcat3')
            upicnv3 = tf.image.resize_bilinear(xconcat3, [size[0], size[1]])

            aspp_up4 = atrous_deep(upicnv3, "ASPP_up4", [3,7,11], depth=16, reuse=reuse)
            #concat4 = tf.concat([upicnv3, aspp_up4], axis=3)
            #upicnv3 = slim.conv2d(upicnv3, 16, [1, 1])

            #xconcat4 = slim.batch_norm(tf.multiply(upicnv3, aspp_up4))
            xconcat4 = tf.concat([upicnv3, aspp_up4], axis=3)

            #for output
            disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

def disp_aspp_u(inputs, args, is_training, reuse, size):
    out_depth = 256
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon,
                                                      activation_fn=tf.nn.elu)):
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
            #net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net_size = tf.shape(net)[1:3]
            # resize the preact features
            #upnet1 = tf.image.resize_bilinear(net, [size[0]//8, size[1]//8])
            upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
            upnet1 = slim.conv2d(upnet1, 512, [1, 1])
            upnet1 = slim.conv2d_transpose(upnet1, 512, [3, 3], stride=2)
            aspp_up1 = atrous_deep(upnet1, "ASPP_up1", depth=256, reuse=reuse)
            #concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            upnet1 = slim.conv2d(upnet1, 256, [1, 1])
            #concat1 = slim.batch_norm(tf.multiply(upnet1, aspp_up1))
            concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            #icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')
            #block3 = slim.conv2d(end_points[args.resnet_model + '/block3'], 128, [1, 1])
            #icnv1 = tf.concat([concat1, block3], axis=3)
            #icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')

            #upsample2 before bn and activation
            #upnet2 = tf.image.resize_bilinear(concat1, [size[0]//4, size[1]//4])
            #upnet2 = slim.batch_norm(upnet2, activation_fn=tf.nn.elu)
            upnet2 = slim.conv2d(concat1, 256, [1, 1])
            upnet2 = slim.conv2d_transpose(upnet2, 256, [3, 3], stride=2)

            aspp_up2 = atrous_deep(upnet2, "ASPP_up2", depth=128, reuse=reuse)
            upnet2 = slim.conv2d(upnet2, 64, [1, 1])
            skip2 = slim.batch_norm(end_points[args.resnet_model + '/block2'], activation_fn=tf.nn.elu)
            skip2 = slim.conv2d(skip2, 64, [1, 1])
            skip2 = slim.conv2d_transpose(skip2, 64, [3, 3], stride=2)

            #xconcat2 = slim.batch_norm(tf.multiply(upicnv1, aspp_up2))
            xconcat2 = tf.concat([upnet2, aspp_up2, skip2], axis=3)
            #block2 = slim.conv2d(end_points[args.resnet_model + '/block2'], 64, [1,1])
            #for output
            #xconcat2_act = slim.batch_norm(xconcat2, activation_fn=tf.nn.elu)
            disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            #upnet3 = tf.image.resize_bilinear(xconcat2, [size[0]//2, size[1]//2])
            #upnet3 = slim.batch_norm(upnet3, activation_fn=tf.nn.elu)
            upnet3 = slim.conv2d(xconcat2, 128, [1, 1])
            upnet3 = slim.conv2d_transpose(upnet3, 128, [3, 3], stride=2)

            aspp_up3 = atrous_deep(upnet3, "ASPP_up3", [3,7,11], depth=64, reuse=reuse)
            upnet3 = slim.conv2d(upnet3, 64, [1, 1])
            skip3 = end_points[args.resnet_model+'/block1']
            skip3 = slim.batch_norm(skip3, activation_fn=tf.nn.elu)
            skip3 = slim.conv2d(skip3, 32, [1, 1])
            skip3 = slim.conv2d_transpose(skip3, 32, [3, 3], stride=2)
            #xconcat3 = slim.batch_norm(tf.multiply(upicnv2, aspp_up3))
            xconcat3 = tf.concat([upnet3, aspp_up3, skip3], axis=3)
            #block1 = slim.conv2d(end_points[args.resnet_model + '/block1'], 32, [1, 1])

            #for output
            #xconcat3_act = slim.batch_norm(xconcat3, activation_fn=tf.nn.elu)
            disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            #upnet4 = tf.image.resize_bilinear(xconcat3, [size[0], size[1]])
            #upnet4 = slim.batch_norm(upnet4, activation_fn=tf.nn.elu)
            upnet4 = slim.conv2d(xconcat3, 64, [1, 1])
            upnet4 = slim.conv2d_transpose(upnet4, 64, [3, 3], stride=2)

            aspp_up4 = atrous_deep(upnet4, "ASPP_up4", [3,7,11], depth=32, reuse=reuse)
            upnet4 = slim.conv2d(upnet4, 32, [1, 1])
            skip4 = end_points[args.resnet_model+'/root_block']
            skip4 = slim.batch_norm(skip4, activation_fn=tf.nn.elu)
            skip4 = slim.conv2d(skip4, 16, [1, 1])
            skip4 = slim.conv2d_transpose(skip4, 16, [3, 3], stride=2)

            xconcat4 = tf.concat([upnet4, aspp_up4, skip4], axis=3)

            #xconcat4 = slim.batch_norm(xconcat4, activation_fn=tf.nn.elu)

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
        #resnet = getattr(resnet_v2, args.resnet_model)
        args.resnet_model = "resnet_v2_50"
        _, end_points = resnet_v2.resnet_v2_50_mod(inputs,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=32,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):
            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']
            #net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net_size = tf.shape(net)[1:3]
            # resize the preact features
            #upnet1 = tf.image.resize_bilinear(net, [size[0]//8, size[1]//8])
            upnet1 = slim.batch_norm(net, activation_fn=tf.nn.elu)
            upnet1 = slim.conv2d(upnet1, 512, [1, 1], activation_fn=None)
            upnet1 = slim.conv2d_transpose(upnet1, 512, [3, 3], stride=2)
            aspp_up1 = atrous_deep(upnet1, "ASPP_up1", depth=256, reuse=reuse)
            #concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            upnet1 = slim.conv2d(upnet1, 256, [1, 1], activation_fn=None)

            upnet1 = slim.conv2d_transpose(tf.concat([upnet1, aspp_up1], axis=3), 512, [3, 3], stride=2)

            skip1 = slim.batch_norm(end_points[args.resnet_model + '/block3'], activation_fn=tf.nn.elu)
            skip1 = slim.conv2d(skip1, 128, [1, 1], activation_fn=None)
            skip1 = slim.conv2d_transpose(skip1, 128, [3, 3], stride=2)

            #concat1 = slim.batch_norm(tf.multiply(upnet1, aspp_up1))
            concat1 = tf.concat([upnet1, skip1], axis=3)
            #icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')
            #block3 = slim.conv2d(end_points[args.resnet_model + '/block3'], 128, [1, 1])
            #icnv1 = tf.concat([concat1, block3], axis=3)
            #icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')

            #upsample2 before bn and activation
            #upnet2 = tf.image.resize_bilinear(concat1, [size[0]//4, size[1]//4])
            #upnet2 = slim.batch_norm(upnet2, activation_fn=tf.nn.elu)
            upnet2 = slim.conv2d(concat1, 256, [1, 1], activation_fn=None)
            upnet2 = slim.conv2d_transpose(upnet2, 256, [3, 3], stride=2)

            aspp_up2 = atrous_deep(upnet2, "ASPP_up2", depth=128, reuse=reuse)
            upnet2 = slim.conv2d(upnet2, 128, [1, 1], activation_fn=None)
            skip2 = slim.batch_norm(end_points[args.resnet_model + '/block2'], activation_fn=tf.nn.elu)
            skip2 = slim.conv2d(skip2, 64, [1, 1], activation_fn=None)
            skip2 = slim.conv2d_transpose(skip2, 64, [3, 3], stride=2)

            #xconcat2 = slim.batch_norm(tf.multiply(upicnv1, aspp_up2))
            xconcat2 = tf.concat([upnet2, aspp_up2, skip2], axis=3)
            #block2 = slim.conv2d(end_points[args.resnet_model + '/block2'], 64, [1,1])
            #for output
            #xconcat2_act = slim.batch_norm(xconcat2, activation_fn=tf.nn.elu)
            disp2 = DISP_SCALING*slim.conv2d(xconcat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            #upnet3 = tf.image.resize_bilinear(xconcat2, [size[0]//2, size[1]//2])
            #upnet3 = slim.batch_norm(upnet3, activation_fn=tf.nn.elu)
            upnet3 = slim.conv2d(xconcat2, 128, [1, 1], activation_fn=None)
            upnet3 = slim.conv2d_transpose(upnet3, 128, [3, 3], stride=2)

            aspp_up3 = atrous_deep(upnet3, "ASPP_up3", [3,7,11], depth=64, reuse=reuse)
            upnet3 = slim.conv2d(upnet3, 64, [1, 1], activation_fn=None)
            skip3 = end_points[args.resnet_model+'/block1']
            skip3 = slim.batch_norm(skip3, activation_fn=tf.nn.elu)
            skip3 = slim.conv2d(skip3, 32, [1, 1], activation_fn=None)
            skip3 = slim.conv2d_transpose(skip3, 32, [3, 3], stride=2)
            #xconcat3 = slim.batch_norm(tf.multiply(upicnv2, aspp_up3))
            xconcat3 = tf.concat([upnet3, aspp_up3, skip3], axis=3)
            #block1 = slim.conv2d(end_points[args.resnet_model + '/block1'], 32, [1, 1])

            #for output
            #xconcat3_act = slim.batch_norm(xconcat3, activation_fn=tf.nn.elu)
            disp3 = DISP_SCALING*slim.conv2d(xconcat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            #upnet4 = tf.image.resize_bilinear(xconcat3, [size[0], size[1]])
            #upnet4 = slim.batch_norm(upnet4, activation_fn=tf.nn.elu)
            upnet4 = slim.conv2d(xconcat3, 64, [1, 1], activation_fn=None)
            upnet4 = slim.conv2d_transpose(upnet4, 64, [3, 3], stride=2)

            aspp_up4 = atrous_deep(upnet4, "ASPP_up4", [3,7,11], depth=32, reuse=reuse)
            upnet4 = slim.conv2d(upnet4, 32, [1, 1], activation_fn=None)
            skip4 = end_points[args.resnet_model+'/root_block']
            skip4 = slim.batch_norm(skip4, activation_fn=tf.nn.elu)
            skip4 = slim.conv2d(skip4, 16, [1, 1], activation_fn=None)
            skip4 = slim.conv2d_transpose(skip4, 16, [3, 3], stride=2)

            xconcat4 = tf.concat([upnet4, aspp_up4, skip4], axis=3)

            #xconcat4 = slim.batch_norm(xconcat4, activation_fn=tf.nn.elu)

            #for output
            disp4  = DISP_SCALING * slim.conv2d(xconcat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points

