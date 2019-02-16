import tensorflow as tf
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
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

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

@slim.add_arg_scope
def atrous_deep(net, scope, rates=[2,3,5], depth=256, reuse=None, activation_fn=tf.nn.elu):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        # apply global average pooling

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rates[0], activation_fn=activation_fn)

        at_pool3x3_2 = slim.conv2d(at_pool3x3_1, depth, [3, 3], scope="conv_3x3_2", rate=rates[1], activation_fn=activation_fn)

        at_pool3x3_3 = slim.conv2d(at_pool3x3_2, depth, [3, 3], scope="conv_3x3_3", rate=rates[2], activation_fn=activation_fn)

        return at_pool3x3_3

def deeplab_v3(inputs, args, is_training, reuse):

    # mean subtraction normalization
    inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]

    # inputs has shape - Original: [batch, 513, 513, 3]
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon)):
        resnet = getattr(resnet_v2, args.resnet_model)
        _, end_points = resnet(inputs,
                               args.number_of_classes,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf.variable_scope("DeepLab_v3", reuse=reuse):

            # get block 4 feature outputs
            net = end_points[args.resnet_model + '/block4']

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')

            size = tf.shape(inputs)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)
            net = tf.image.resize_bilinear(net, size)
            return net


def disp(inputs, args, is_training, reuse):

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

            #net_b_pool = slim.conv2d(net, out_depth, [1, 1], activation_fn=tf.nn.relu,
            #                  normalizer_fn=None, scope='b_pool_conv')

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net = slim.conv2d(net, out_depth, [1, 1], activation_fn=tf.nn.relu,
                              normalizer_fn=None, scope='logits')

            #net = tf.concat([net_b_pool, net], axis=3)
            size = tf.shape(inputs)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)

            #net = tf.image.resize_bilinear(net, size)

            upcnv1 = slim.conv2d_transpose(net, 128,  [3, 3], stride=2, scope='upcnv1')
            #i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(upcnv1, 128,  [3, 3], stride=1, scope='icnv1')

            upcnv2 = slim.conv2d_transpose(icnv1, 64, [3, 3], stride=2, scope='upcnv2')
            icnv2 = slim.conv2d(upcnv2, 64, [3, 3], stride=1, scope='icnv2')

            upcnv3 = slim.conv2d_transpose(icnv2, 32, [3, 3], stride=2, scope='upcnv3')
            icnv3 = slim.conv2d(upcnv3, 32, [3, 3], stride=1, scope='icnv3')

            #upcnv4 = slim.conv2d_transpose(icnv3, 16, [3, 3], stride=2, scope='upcnv4')
            #icnv4 = slim.conv2d(upcnv4, 16, [3, 3], stride=1, scope='icnv4')

            disp1  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP

            return [disp1], end_points

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

            #net_b_pool = slim.conv2d(net, out_depth, [1, 1], activation_fn=tf.nn.relu,
            #                  scope='b_pool_conv')

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", depth=256, reuse=reuse)

            net = slim.conv2d(net, out_depth, [1, 1], activation_fn=tf.nn.relu,
                              scope='logits')

            #net = tf.concat([net_b_pool, net], axis=3)
            size = tf.shape(inputs)[1:3]
            net_size = tf.shape(net)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)

            #net = tf.image.resize_bilinear(net, size)

            upnet1 = tf.image.resize_bilinear(net, [net_size[0]*2, net_size[1]*2])
            sccnv1 = slim.conv2d(upnet1, 128, [3, 3], activation_fn=None, stride=1, scope='sccnv1')

            upcnv1 = slim.conv2d_transpose(net, 128,  [3, 3], stride=2, scope='upcnv1')
            #i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = sccnv1 + slim.conv2d(upcnv1, 128,  [3, 3], activation_fn=None, stride=1, scope='icnv1')


            #disp4 = slim.conv2d(icnv1, 1, [3, 3], stride=1, scope='disp4')

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


def disp_aspp(inputs, args, is_training, reuse, size):

    # mean subtraction normalization
    #inputs = inputs - [_R_MEAN, _G_MEAN, _B_MEAN]
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

            #net = slim.conv2d(net, out_depth, [1, 1], activation_fn=tf.nn.elu,
            #                  scope='logits')

            #size = tf.shape(inputs)[1:3]
            net_size = tf.shape(net)[1:3]
            # resize the output logits to match the labels dimensions
            #net = tf.image.resize_nearest_neighbor(net, size)

            #a residual block for up sampling
            upnet1 = tf.image.resize_bilinear(net, [size[0]//8, size[1]//8])
            aspp_up1 = atrous_deep(upnet1, "ASPP_up1", depth=128, reuse=reuse)
            concat1 = tf.concat([upnet1, aspp_up1], axis=3)
            icnv1 = slim.conv2d(concat1, 128, [3,3], scope='icnv1')

            #sccnv1 = slim.conv2d(upnet1, 128, [3, 3], stride=1, scope='sccnv1')

            #upcnv1 = slim.conv2d_transpose(net, 128,  [3, 3], stride=2, scope='upcnv1')
            #icnv1  = tf.nn.elu(sccnv1 + slim.conv2d(upcnv1, 128,  [3, 3], activation_fn=tf.nn.elu, stride=1, scope='icnv1'))


            #upsample2
            upicnv1 = tf.image.resize_bilinear(icnv1, [size[0]//4, size[1]//4])

            aspp_up2 = atrous_deep(upicnv1, "ASPP_up2", depth=64, reuse=reuse)
            #aspp_up2 = slim.conv2d(upicnv1, depth, [3, 3], scope="ASPP_up2", rate=2)
            #aspp_up25 = slim.conv2d(aspp_up2, depth, [3, 3], scope="ASPP_up25",rate=5)
            concat2 = tf.concat([upicnv1, aspp_up2], axis=3)

            #for output
            disp2 = DISP_SCALING*slim.conv2d(concat2, 1, [3, 3], activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP

            #upsampe3
            upicnv2 = tf.image.resize_bilinear(aspp_up2, [size[0]//2, size[1]//2])

            aspp_up3 = atrous_deep(upicnv2, "ASPP_up3", [3,5,7], depth=32, reuse=reuse)
            concat3 = tf.concat([upicnv2, aspp_up3], axis=3)

            #for output
            disp3 = DISP_SCALING*slim.conv2d(concat3, 1, [3, 3], stride=1, activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP

            #upsample4
            upicnv3 = tf.image.resize_bilinear(aspp_up3, [size[0], size[1]])

            aspp_up4 = atrous_deep(upicnv3, "ASPP_up4", [3,7,11], depth=16, reuse=reuse)
            concat4 = tf.concat([upicnv3, aspp_up4], axis=3)

            #for output
            disp4  = DISP_SCALING * slim.conv2d(concat4, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP

            return [disp4, disp3, disp2], end_points
