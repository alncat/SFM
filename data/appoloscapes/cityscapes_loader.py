from __future__ import division
import json
import os
import numpy as np
import scipy.misc
from glob import glob

class cityscapes_loader(object):
    def __init__(self,
                 dataset_dir,
                 split='Col',
                 crop_bottom=True, # Get rid of the car logo
                 sample_gap=1,  # Sample every two frames to match KITTI frame rate
                 img_height=171,
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.split = split
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = crop_bottom
        self.sample_gap = sample_gap
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        assert seq_length % 2 != 0, 'seq_length must be odd!'
        self.frames = self.collect_frames(split)
        self.num_frames = len(self.frames)
        if split == 'train':
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print('Total frames collected: %d' % self.num_frames)

    def collect_frames(self, split):
        img_dir = self.dataset_dir + '/'
        city_list = os.listdir(img_dir)
        frames = []
        for city in city_list:
            city_img_dir = img_dir + city + '/ColorImage/'
            record_list = os.listdir(city_img_dir)
            for record in record_list:
                camera_list = os.listdir(city_img_dir + record)
                cameras_imgs = []
                for camera in camera_list:
                    camera_dir = city_img_dir + record + '/' + camera
                    img_files = glob(camera_dir + '/*.jpg')
                    camera_imgs = []
                    for f in img_files:
                        #frame_id = os.path.basename(f).split('_')[0]
                        camera_imgs.append(f)
                    camera_imgs.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
                    cameras_imgs.append(camera_imgs)
                frames.append(cameras_imgs)
        return frames

    def get_train_example_with_idx(self, tgt_idx, cam_idx, frame_id):
        #if not self.is_valid_example(tgt_frame_id):
        #    return False
        if frame_id == 0 or frame_id == len(self.frames[tgt_idx][cam_idx]):
            return False
        example = self.load_example(tgt_idx, cam_idx, frame_id)
        return example

    def load_intrinsics(self, cam_idx):
        if cam_idx == 0:
            fx=2304.54786556982
            fy=2305.875668062
            u0=1686.23787612802
            v0=1354.98486439791
        else:
            fx=2300.39065314361
            fy=2301.31478860597
            u0=1713.21615190657
            v0=1342.91100799715
        intrinsics = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0,  0,  1]])
        return intrinsics

    def is_valid_example(self, tgt_frame_id):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence',
                                self.split, city, curr_frame_id + 'leftImg8bit.png')
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_idx, cam_idx, tgt_idx, seq_length, crop_bottom):
        tgt_frame_id = self.frame[tgt_idx][cam_idx][tgt_idx]
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_image_file = self.frame[tgt_idx][cam_idx][tgt_idx+o]
            curr_img = scipy.misc.imread(curr_image_file)
            raw_shape = np.copy(curr_img.shape)
            if o == 0:
                zoom_y = self.img_height/raw_shape[0]
                zoom_x = self.img_width/raw_shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            if crop_bottom:
                ymax = int(curr_img.shape[0] * 0.75)
                curr_img = curr_img[:ymax]
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, tgt_idx, cam_idx, frame_id, load_gt_pose=False):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_idx, cam_idx, frame_id, self.seq_length, self.crop_bottom)
        intrinsics = self.load_intrinsics(cam_idx)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_frame_id.split('_')[0]
        example['file_name'] = tgt_frame_id[:-1]
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out
