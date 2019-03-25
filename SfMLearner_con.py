from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
import network
from utils import *

class Dummy:
    pass
args = Dummy()
args.l2_regularizer = 0.0001
args.is_training = True
args.batch_norm_decay = 0.9997
args.batch_norm_epsilon = 1e-5
args.output_stride = 16
args.resnet_model = "resnet_v2_50"

class SfMLearner(object):
    def __init__(self):
        self.use_cspn = False
        pass

    def build_train_graph(self, tgt_image, src_image_stack, intrinsics, is_training = True, reuse=False):
        opt = self.opt
        #with tf.name_scope("data_loading", reuse=reuse):
            #if is_training:
            #    tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            #else:
            #    tgt_image, src_image_stack, intrinsics = loader.load_val_batch()
            #tgt_image, src_image_stack, intrinsics = loader()
        tgt_image = self.preprocess_image(tgt_image, is_training)
        src_image_stack = self.preprocess_image(src_image_stack, is_training)

        #with tf.name_scope("depth_prediction"):
        if not self.use_cspn:
            pred_disp, depth_net_endpoints = disp_aspp_u_dense(tgt_image,
                                                  args, is_training, reuse, [opt.img_height, opt.img_width])
            pred_disp = [d/tf.reduce_mean(d, axis=[1,2,3], keep_dims=True) for d in pred_disp]
            pred_depth = [1./d for d in pred_disp]
        else:
            pred_disp, depth_net_endpoints = disp_net_cspn(tgt_image,
                                                  is_training=is_training)
            pred_depth = pred_disp

        #with tf.name_scope("pose_and_explainability_prediction", reuse=reuse):
        pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
                pose_trans_net(tgt_image,
                             src_image_stack,
                             do_trans=(opt.explain_reg_weight > 0),
                             is_training=is_training,
                             reuse = reuse)

        #with tf.name_scope("compute_loss", reuse=reuse):
        pixel_loss = 0
        exp_loss = 0
        smooth_loss = 0
        tgt_image_all = []
        src_image_stack_all = []
        proj_image_stack_all = []
        proj_error_stack_all = []
        proj_src_image_stack_all = []
        exp_mask_stack_all = []
        alpha = 0.15
        for s in range(opt.num_scales):
            #if opt.explain_reg_weight > 0:
                # Construct a reference explainability mask (i.e. all
                # pixels are explainable)
            #    ref_exp_mask = self.get_reference_explain_mask(s)
            # Scale the source and target images for computing loss at the
            # according scale.
            #curr_tgt_image = tf.image.resize_area(tgt_image,
            #    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
            curr_tgt_image = tgt_image
            curr_src_image_stack = src_image_stack
            #curr_src_image_stack = tf.image.resize_area(src_image_stack,
            #    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

            pred_depth_s = tf.image.resize_bilinear(pred_depth[s], [opt.img_height, opt.img_width])

            if opt.smooth_weight > 0 and not self.use_cspn:
                smooth_loss += opt.smooth_weight/(2**s) * \
                    self.compute_smooth_loss(pred_disp[s])
                if opt.explain_reg_weight > 0:
                    smooth_loss += opt.explain_reg_weight/(2**s) * \
                    self.compute_smooth_loss(pred_exp_logits[s])


            for i in range(opt.num_source):
                # Inverse warp the source image to the target image frame
                #concat depth to src image
                if opt.explain_reg_weight > 0:
                    curr_exp_logits = tf.slice(pred_exp_logits[s],
                                                [0, 0, 0, i],
                                                [-1, -1, -1, 1])
                    curr_exp_logits = tf.image.resize_bilinear(curr_exp_logits, [opt.img_height, opt.img_width])
                    pred_depth_s += curr_exp_logits
                    #normalize depth after adding
                    #pred_depth_s = pred_depth_s/tf.reduce_mean(pred_depth_s, axis=[1,2,3], keep_dims=True)

                curr_proj_image, proj_mask  = projective_inverse_warp(
                    curr_src_image_stack[:,:,:,3*i:3*(i+1)],
                    #curr_tgt_image,
                    tf.squeeze(pred_depth_s, axis=3),
                    pred_poses[:,i,:],
                    intrinsics[:,0,:,:])
                curr_proj_src_image, proj_src_mask = projective_warp(
                    curr_tgt_image,
                    tf.squeeze(pred_depth_s, axis=3),
                    pred_poses[:,i,:],
                    intrinsics[:,0,:,:])
                curr_proj_error = proj_mask*tf.abs(curr_proj_image - curr_tgt_image)
                curr_proj_error += proj_src_mask*tf.abs(curr_proj_src_image - curr_src_image_stack[:,:,:,3*i:3*(i+1)])
                #curr_proj_error = tf.abs(curr_proj_image - curr_src_image_stack[:,:,:,3*i:3*(i+1)])
                curr_ssim_error = proj_mask*tf_ssim(curr_proj_image, curr_tgt_image, mean_metric=False)
                curr_ssim_error += proj_src_mask*tf_ssim(curr_proj_src_image, curr_src_image_stack[:,:,:,3*i:3*(i+1)])
                # Cross-entropy loss as regularization for the
                # explainability prediction

                # Photo-consistency loss weighted by explainability
                pixel_loss += alpha*tf.reduce_mean(curr_proj_error) + (1-alpha)*tf.reduce_mean(curr_ssim_error)
                # Prepare images for tensorboard summaries
                if i == 0:
                    proj_image_stack = curr_proj_image
                    proj_src_image_stack = curr_proj_src_image
                    proj_error_stack = curr_proj_error
                    if opt.explain_reg_weight > 0:
                        exp_mask_stack = tf.expand_dims(curr_exp_logits[:,:,:,0], -1)
                else:
                    proj_image_stack = tf.concat([proj_image_stack,
                                                  curr_proj_image], axis=3)
                    proj_src_image_stack = tf.concat([proj_src_image_stack,
                                                  curr_proj_src_image], axis=3)
                    proj_error_stack = tf.concat([proj_error_stack,
                                                  curr_proj_error], axis=3)
                    if opt.explain_reg_weight > 0:
                        exp_mask_stack = tf.concat([exp_mask_stack,
                            tf.expand_dims(curr_exp_logits[:,:,:,0], -1)], axis=3)
            tgt_image_all.append(curr_tgt_image)
            src_image_stack_all.append(curr_src_image_stack)
            proj_image_stack_all.append(proj_image_stack)
            proj_src_image_stack_all.append(proj_src_image_stack)
            proj_error_stack_all.append(proj_error_stack)
            if opt.explain_reg_weight > 0:
                exp_mask_stack_all.append(exp_mask_stack)
        total_loss = pixel_loss + smooth_loss + exp_loss


        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.pred_poses = pred_poses
        self.total_loss = total_loss
        self.pixel_loss = pixel_loss
        self.exp_loss = exp_loss
        self.smooth_loss = smooth_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_src_image_stack_all = proj_src_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all
        return self.total_loss

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (opt.batch_size,
                                int(opt.img_height/(2**downscaling)),
                                int(opt.img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def compute_exp_reg_loss(self, pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def compute_smooth_loss(self, pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        beta = 0.25
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        gx = tf.reduce_mean(tf.abs(dx)) + tf.reduce_mean(tf.abs(dy))
        hx  =  tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))
        return beta*gx + (1-beta)*hx

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_loss", self.pixel_loss)
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        for s in range(1):
            tf.summary.histogram("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_depth_image' % s, 1./self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(opt.num_source):
                if opt.explain_reg_weight > 0:
                    tf.summary.histogram("scaled%d_exp_mask%d" % (s, i), self.exp_mask_stack_all[s][:,:,:,i])
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:,:,:,i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i),
                    self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_projected_src_image_%d'% (s, i),
                    self.deprocess_image(self.proj_src_image_stack_all[s][:, :, :, i*3:(i+1)*3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    self.deprocess_image(tf.clip_by_value(self.proj_error_stack_all[s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)))
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 3
        self.opt = opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)

        is_training_tf = tf.placeholder(tf.bool, shape=[])
        #select from a queue
        #select_q = tf.placeholder(tf.int32, name="select_q", shape=[])
        with tf.name_scope("train_data"):
            tgt_image_train, src_image_stack_train, intrinsics_train = loader.load_train_batch()
            tgt_image_train, src_image_stack_train, intrinsics_train = loader.fetch_train_batch(tgt_image_train, src_image_stack_train, intrinsics_train)

        with tf.name_scope("validation_data"):
            tgt_image_val, src_image_stack_val, intrinsics_val = loader.load_val_batch()
            tgt_image_val, src_image_stack_val, intrinsics_val = loader.fetch_val_batch(tgt_image_val, src_image_stack_val, intrinsics_val)

        #tgt_image = tf.cond(is_training_tf, lambda: tgt_image_train, lambda: tgt_image_val)
        #src_image_stack = tf.cond(is_training_tf, lambda: src_image_stack_train, lambda: src_image_stack_val)
        #intrinsics = tf.cond(is_training_tf, lambda: intrinsics_train, lambda: intrinsics_val)

        total_loss = tf.cond(is_training_tf, true_fn = lambda: self.build_train_graph(
                tgt_image_train, src_image_stack_train, intrinsics_train, is_training=True, reuse=False),
                false_fn=lambda: self.build_train_graph(tgt_image_val, src_image_stack_val, intrinsics_val, is_training=False, reuse=True))
        #total_loss  = self.build_train_graph(
        #        tgt_image_train, src_image_stack_train, intrinsics_train, is_training=True, reuse=False)
        self.steps_per_epoch = loader.steps_per_epoch
        self.collect_summaries()
        #using polyak averaging
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9997)
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])


        #create train op
        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()]
            self.global_step = tf.Variable(0,
                                           name='global_step',
                                           trainable=False)
            #learning_rate = tf.train.inverse_time_decay(opt.learning_rate, self.global_step, 25000., 1)
            learning_rate = opt.learning_rate
            #optim = tf.train.AdamOptimizer(learning_rate, opt.beta1)
            optim = tf.contrib.opt.NadamOptimizer(learning_rate)
            # self.grads_and_vars = optim.compute_gradients(total_loss,
            #                                               var_list=train_vars)
            # self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.opt_op = slim.learning.create_train_op(total_loss, optim, global_step=self.global_step)
            with tf.control_dependencies([self.opt_op]):
                self.train_op = self.ema.apply(tf.trainable_variables())
            #self.incr_global_step = tf.assign(self.global_step,
            #                                  self.global_step+1)
        #variables_to_restore = slim.get_variables_to_restore(exclude=[self.ema.average_name(var) for var in tf.trainable_variables()])

        #variables_to_restore = slim.get_variables_to_restore(exclude=[args.resnet_model + "/logits", "optimizer_vars",
        #                                                      "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"] + \
        #                                                      ["DeepLab_v3/icnv"+str(i+1) for i in range(4)] + \
        #                                                      ["DeepLab_v3/upcnv"+str(i+1) for i in range(4)] + \
        #                                                      ["DeepLab_v3/disp1"] +\
        #                                                      ["pose_exp_net/cnv"+str(i+1) for i in range(7)] + \
        #                                                      ["pose_exp_net/pose/pred", "pose_exp_net/pose/cnv6", "pose_exp_net/pose/cnv7"])
        #self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
        #self.restorer = tf.train.Saver([var for var in variables_to_restore] ,\
        #                            #[self.global_step],
        #                             max_to_keep=10)
        self.saver = tf.train.Saver(self.ema.variables_to_restore())
        #variables_to_restore = set(tf.global_variables() + tf.local_variables())
        #for var in tf.trainable_variables():
        #    if self.ema.average_name(var) in variables_to_restore:
        #        variables_to_restore.remove(self.ema.average_name(var))
        #for var in tf.moving_average_variables():
        #    if var in variables_to_restore:
        #        variables_to_restore.remove(var)

        #self.saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto(device_count = {'GPU': 2})
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            #sess.run(tf.local_variables_initializer())
            #sess.run(tf.global_variables_initializer())

            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
            #try:
                self.saver.restore(sess, checkpoint)

            #    #self.restorer.restore(sess, "./resnet/checkpoints/" + args.resnet_model + ".ckpt")
            #    self.restorer.restore(sess, "./resnet/checkpoints/model.ckpt")
            #    print("Model checkpoints for " + args.resnet_mode + " restored!")
            #except FileNotFoundError:
            #    print("ResNet checkpoints not found.")

            start_time = time.time()
            average_train_loss = 0.
            for step in range(1, opt.max_steps):
                fetches = {
                        #"tgt": tgt_image,
                        #"src": src_image_stack,
                        #"ins": intrinsics,
                        "train": self.train_op,
                        "loss": total_loss,
                        "global_step": self.global_step,
                        #"incr_global_step": self.incr_global_step
                }
                feed_dict = {is_training_tf: True}

                #if step % opt.validation_freq == 0:
                #    fetches["loss"] = total_loss
                #    fetches["summary"] = sv.summary_op

                results = sess.run(fetches, feed_dict)
                gs = results["global_step"]
                average_train_loss += results["loss"]

                if step % opt.validation_freq == 0:

                    average_val_loss = 0.
                    val_steps = opt.validation_freq//10
                    for j in range(val_steps):
                        fetches = {"val": total_loss}
                        if j % 4 == 0:
                            fetches["summary"] = sv.summary_op
                        feed_dict = {is_training_tf: False}
                        val_results = sess.run(fetches, feed_dict)
                        average_val_loss += val_results["val"]
                        if j % 4 == 0:
                            sv.summary_writer.add_summary(val_results["summary"], gs+j)
                    average_val_loss /= val_steps


                if step % opt.validation_freq == 0:
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    average_train_loss /= opt.validation_freq
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f val_loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq,
                                average_train_loss, average_val_loss))
                    start_time = time.time()
                    average_train_loss = 0.

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % (self.steps_per_epoch//3) == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                    self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            if self.use_cspn:
                pred_depth, depth_net_endpoints = disp_net_cspn(
                        input_mc, is_training=False)
            else:
                pred_disp, depth_net_endpoints = network.disp_aspp_u(
                    input_mc, args, False, False, [self.img_height, self.img_width])
                pred_depth = [1./disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth
        self.depth_epts = depth_net_endpoints

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
            self.img_height, self.img_width * self.seq_length, 3],
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image, is_training=False):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #if is_training:
        #    image = tf.image.random_brightness(image, 0.2)
        #    image = tf.clip_by_value(image, 0., 1.)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        #self.saver = tf.train.Saver(self.ema.variables_to_restore())
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
