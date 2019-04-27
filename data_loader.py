from __future__ import division
import os
import math
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self,
                 dataset_dir=None,
                 batch_size=None,
                 img_height=None,
                 img_width=None,
                 num_source=None,
                 num_scales=None):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_source = num_source
        self.num_scales = num_scales

    def load_val_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'val')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'],
            seed=seed,
            shuffle=True)
        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'],
            seed=seed,
            shuffle=True)
        #self.steps_per_epoch = int(
        #    len(file_list['image_file_list'])//self.batch_size)

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])
        return tgt_image, src_image_stack, intrinsics

        # Form training batches
    def fetch_val_batch(self, tgt_image, src_image_stack, intrinsics):
        src_image_stack, tgt_image, intrinsics = \
                tf.train.batch([src_image_stack, tgt_image, intrinsics],
                               batch_size=self.batch_size)

        # Data augmentation
        #image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        #image_all, intrinsics = self.data_augmentation(
        #    image_all, intrinsics, self.img_height, self.img_width)
        #tgt_image = image_all[:, :, :, :3]
        #src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        return tgt_image, src_image_stack, intrinsics

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list = self.format_file_list(self.dataset_dir, 'train')
        image_paths_queue = tf.train.string_input_producer(
            file_list['image_file_list'],
            seed=seed,
            shuffle=True)
        cam_paths_queue = tf.train.string_input_producer(
            file_list['cam_file_list'],
            seed=seed,
            shuffle=True)
        self.steps_per_epoch = int(
            len(file_list['image_file_list'])//self.batch_size)

        # Load the list of validation files into queues
        #val_file_list = self.format_file_list(self.dataset_dir, 'val')
        #image_paths_val_queue = tf.train.string_input_producer(
        #    val_file_list['image_file_list'],
        #    seed=seed,
        #    shuffle=True)
        #cam_paths_val_queue = tf.train.string_input_producer(
        #    val_file_list['cam_file_list'],
        #    seed=seed,
        #    shuffle=True)

        #image_paths_queue = tf.QueueBase.from_list(select_q, [image_paths_train_queue, image_paths_val_queue])
        #cam_paths_queue = tf.QueueBase.from_list(select_q, [cam_paths_train_queue, cam_paths_val_queue])

        # Load images
        img_reader = tf.WholeFileReader()
        _, image_contents = img_reader.read(image_paths_queue)
        image_seq = tf.image.decode_jpeg(image_contents)
        tgt_image, src_image_stack = \
            self.unpack_image_sequence(
                image_seq, self.img_height, self.img_width, self.num_source)

        # Load camera intrinsics
        cam_reader = tf.TextLineReader()
        _, raw_cam_contents = cam_reader.read(cam_paths_queue)
        rec_def = []
        for i in range(9):
            rec_def.append([1.])
        raw_cam_vec = tf.decode_csv(raw_cam_contents,
                                    record_defaults=rec_def)
        raw_cam_vec = tf.stack(raw_cam_vec)
        intrinsics = tf.reshape(raw_cam_vec, [3, 3])
        return tgt_image, src_image_stack, intrinsics

    def fetch_train_batch(self, tgt_image, src_image_stack, intrinsics):

        # Form training batches
        src_image_stack, tgt_image, intrinsics = \
                tf.train.batch([src_image_stack, tgt_image, intrinsics],
                               batch_size=self.batch_size)

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, self.img_height, self.img_width)
        image_all = tf.image.random_contrast(image_all, 0.5, 1.5)
        #image_all = tf.image.random_saturation(image_all, 0.5, 1.5)
        #image_all = tf.image.random_brightness(image_all, 0.2)
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        return tgt_image, src_image_stack, intrinsics

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # random rotate
        def random_rotate(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            random_angle = tf.random_uniform([], -0.1, 0.1)
            rotate_image = tf.contrib.image.rotate(im, random_angle, interpolation='BILINEAR')

            batch_size, rh, rw, _ = tf.unstack(tf.shape(rotate_image))
            l_x, l_y, lrr_width, lrr_height = self._largest_rotated_rect(in_w, in_h, random_angle, rw, rh)
            l_x = tf.cast(l_x, dtype=tf.int32)
            l_y = tf.cast(l_y, dtype=tf.int32)
            lrr_height = tf.cast(lrr_height, dtype=tf.int32)
            lrr_width = tf.cast(lrr_width, dtype=tf.int32)
            im = tf.image.crop_to_bounding_box(rotate_image, l_y, l_x, lrr_height, lrr_width)
            im = tf.image.resize_bilinear(im, [out_h, out_w])
            lrr_height = tf.cast(lrr_height, dtype=tf.float32)
            lrr_width = tf.cast(lrr_width, dtype=tf.float32)

            scale_x = out_w/lrr_width
            scale_y = out_h/lrr_height

            fx = intrinsics[:,0,0]*scale_x
            fy = intrinsics[:,1,1]*scale_y
            cx = (intrinsics[:,0,2] - (in_w/2. - lrr_width/2.))*scale_x
            cy = (intrinsics[:,1,2] - (in_h/2. - lrr_height/2.))*scale_y
            #cx = cx * tf.cos(random_angle) + cy * tf.sin(random_angle)
            #cy = cx * tf.sin(random_angle) + cy * tf.cos(random_angle)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)

            return im, intrinsics
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h_, in_w_, _ = tf.unstack(tf.shape(im))#im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            in_h = tf.cast(in_h_, dtype=tf.float32)
            in_w = tf.cast(in_w_, dtype=tf.float32)
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_bilinear(im, [out_h, out_w])
            #im = tf.image.resize_bilinear(im, [out_h, in_w_])
            #im = tf.image.crop_to_bounding_box(
            #        im, 0, 0, in_h_, in_w_)
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([], 0, in_h - out_h + 1, dtype=tf.int32)
            offset_x = tf.random_uniform([], 0, in_w - out_w + 1, dtype=tf.int32)
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics
        def random_flipping(im):
            flip = tf.random_uniform([], 0, 1)
            im = tf.case([(tf.less(flip, 0.5), lambda: tf.reverse(im, axis=[1]))], default = lambda: im)
            #im = tf.image.flip_up_down(im)
            return im
        def random_swapping(im):
            flip = tf.random_uniform([], 0, 1)
            im = tf.case([(tf.less(flip, 0.5), lambda: tf.reverse(im, axis=[2]))], default = lambda: im)
            #im = tf.image.flip_up_down(im)
            return im
        def random_saturation(im):
            random_colors = tf.random_uniform([3], 0.8, 1.2)
            white = tf.ones([out_h, out_w])
            color_image = tf.stack([white*random_colors[i] for i in range(3)], axis=2)
            color_image = tf.concat([color_image, color_image, color_image], axis=2)
            im = im * color_image
            return im
        def random_brightness(im):
            random_bright = tf.random_uniform([], 1/1.2, 1/0.8)
            im = im * random_bright
            return im

        def random_gamma(im):
            random_g = tf.random_uniform([], 0.8, 1.2)
            im = im ** random_g
            return im

        def random_hue(im):
            random_h = tf.random_uniform([1,1,1,1], -0.25, 0.25)
            random_s = tf.random_uniform([], 1/1.25, 1.25)
            #zeros = tf.zeros([1,1,1,2])
            #offset = tf.concat([random_h, zeros], axis=3)
            new_ims = []
            for i in range(3):
                new_im = tf.image.rgb_to_hsv(tf.slice(im, [0, 0, 0, i*3], [-1, -1, -1, 3]))
                hue = tf.slice(new_im, [0, 0, 0, 0], [-1, -1, -1, 1])
                saturation = tf.slice(new_im, [0, 0, 0, 1], [-1, -1, -1, 1])
                value = tf.slice(new_im, [0, 0, 0, 2], [-1, -1, -1, 1])

                hue = tf.mod(hue + (random_h + 1.), 1.)
                saturation = tf.clip_by_value(saturation*random_s, 0., 1.)
                new_im = tf.concat([hue, saturation, value], axis=3)
                new_im = tf.image.hsv_to_rgb(new_im)
                new_im = tf.image.convert_image_dtype(new_im, tf.float32)
                new_ims.append(new_im)
            return tf.concat(new_ims, axis=3)
        #im, intrinsics = random_rotate(im, intrinsics)
        #im, intrinsics = random_scaling(im, intrinsics)
        #im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        #im = random_gamma(im)
        im = tf.image.convert_image_dtype(im, tf.float32)
        #im = random_hue(im)
        im = random_brightness(im)
        im = random_saturation(im)
        im = tf.image.convert_image_dtype(im, tf.uint8, saturate=True)
        #im = tf.cast(im, dtype=tf.uint8)
        #im = random_flipping(im)
        im = random_swapping(im)
        return im, intrinsics

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
            frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source//2)), -1])

        #flip = tf.random_uniform([], 0, 1)
        #src_image_seq = tf.case([(tf.less(flip, 0.5), lambda: tf.concat([src_image_1, src_image_2], axis=1))], default = lambda: tf.concat([src_image_2, src_image_1], axis=1))

        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, i*img_width, 0],
                                    [-1, img_width, -1])
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq,
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq,
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        # Stack source frames along the color channels (i.e. [B, H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                    [0, 0, i*img_width, 0],
                                    [-1, -1, img_width, -1])
                                    for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
    def _largest_rotated_rect(self, w, h, angle, rw, rh):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
        Converted to Python by Aaron Snoswell
        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        angle = tf.abs(angle)
        w = tf.cast(w, dtype=tf.float32)
        h = tf.cast(h, dtype=tf.float32)
        rw = tf.cast(rw, dtype=tf.float32)
        rh = tf.cast(rh, dtype=tf.float32)

        cos2a = tf.cos(2*angle)
        bw = (w*tf.cos(angle) - h*tf.sin(angle))/cos2a
        bh = (h*tf.cos(angle) - w*tf.sin(angle))/cos2a
        return (
                (rw - bw)/2,
                (rh - bh)/2,
                bw,
                bh
                )
