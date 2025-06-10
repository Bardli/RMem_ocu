from __future__ import division
import os
from os.path import exists
from glob import glob
import json
import random
import cv2
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset # Updated this line for clarity
import torchvision.transforms as TF

import dataloaders.image_transforms as IT

cv2.setNumThreads(0)


def _get_images(sample):
    return [sample['ref_img'], sample['prev_img']] + sample['curr_img']


def _get_labels(sample):
    return [sample['ref_label'], sample['prev_label']] + sample['curr_label']


def _merge_sample(sample1, sample2, min_obj_pixels=100, max_obj_n=10, ignore_in_merge=False):

    sample1_images = _get_images(sample1)
    sample2_images = _get_images(sample2)

    sample1_labels = _get_labels(sample1)
    sample2_labels = _get_labels(sample2)

    obj_idx = torch.arange(0, max_obj_n * 2 + 1).view(max_obj_n * 2 + 1, 1, 1)
    selected_idx = None
    selected_obj = None

    all_img = []
    all_mask = []
    for idx, (s1_img, s2_img, s1_label, s2_label) in enumerate(
            zip(sample1_images, sample2_images, sample1_labels,
                sample2_labels)):
        s2_fg = (s2_label > 0).float()
        s2_bg = 1 - s2_fg
        merged_img = s1_img * s2_bg + s2_img * s2_fg
        merged_mask = s1_label * s2_bg.long() + (
            (s2_label + max_obj_n) * s2_fg.long())
        merged_mask = (merged_mask == obj_idx).float()
        if idx == 0:
            after_merge_pixels = merged_mask.sum(dim=(1, 2), keepdim=True)
            selected_idx = after_merge_pixels > min_obj_pixels
            selected_idx[0] = True
            obj_num = selected_idx.sum().int().item() - 1
            selected_idx = selected_idx.expand(-1,
                                               s1_label.size()[1],
                                               s1_label.size()[2])
            if obj_num > max_obj_n:
                selected_obj = list(range(1, obj_num + 1))
                random.shuffle(selected_obj)
                selected_obj = [0] + selected_obj[:max_obj_n]

        merged_mask = merged_mask[selected_idx].view(obj_num + 1,
                                                     s1_label.size()[1],
                                                     s1_label.size()[2])
        if obj_num > max_obj_n:
            merged_mask = merged_mask[selected_obj]
        merged_mask[0] += 0.1
        merged_mask = torch.argmax(merged_mask, dim=0, keepdim=True).long()

        if ignore_in_merge:
            merged_mask = merged_mask + (s1_label == 255).long() * 255 * (merged_mask == 0).long()
            merged_mask = merged_mask + (s2_label == 255).long() * 255 * (merged_mask == 0).long()

        all_img.append(merged_img)
        all_mask.append(merged_mask)

    sample = {
        'ref_img': all_img[0],
        'prev_img': all_img[1],
        'curr_img': all_img[2:],
        'ref_label': all_mask[0],
        'prev_label': all_mask[1],
        'curr_label': all_mask[2:]
    }
    sample['meta'] = sample1['meta']
    sample['meta']['obj_num'] = min(obj_num, max_obj_n)
    return sample


class StaticTrain(Dataset):
    def __init__(self,
                 root,
                 output_size,
                 seq_len=5,
                 max_obj_n=10,
                 dynamic_merge=True,
                 merge_prob=1.0):
        self.root = root
        self.clip_n = seq_len
        self.output_size = output_size
        self.max_obj_n = max_obj_n

        self.dynamic_merge = dynamic_merge
        self.merge_prob = merge_prob

        self.img_list = list()
        self.mask_list = list()

        dataset_list = list()
        lines = ['COCO', 'ECSSD', 'MSRA10K', 'PASCAL-S', 'PASCALVOC2012']
        for line in lines:
            dataset_name = line.strip()

            img_dir = os.path.join(root, 'JPEGImages', dataset_name)
            mask_dir = os.path.join(root, 'Annotations', dataset_name)

            img_list = sorted(glob(os.path.join(img_dir, '*.jpg'))) + \
                sorted(glob(os.path.join(img_dir, '*.png')))
            mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

            if len(img_list) > 0:
                if len(img_list) == len(mask_list):
                    dataset_list.append(dataset_name)
                    self.img_list += img_list
                    self.mask_list += mask_list
                    print(f'\t{dataset_name}: {len(img_list)} imgs.')
                else:
                    print(
                        f'\tPreTrain dataset {dataset_name} has {len(img_list)} imgs and {len(mask_list)} annots. Not match! Skip.'
                    )
            else:
                print(
                    f'\tPreTrain dataset {dataset_name} doesn\'t exist. Skip.')

        print(
            f'{len(self.img_list)} imgs are used for PreTrain. They are from {dataset_list}.'
        )

        self.pre_random_horizontal_flip = IT.RandomHorizontalFlip(0.5)

        self.random_horizontal_flip = IT.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        self.random_affine = IT.RandomAffine(degrees=20,
                                             translate=(0.1, 0.1),
                                             scale=(0.9, 1.1),
                                             shear=10,
                                             resample=Image.BICUBIC,
                                             fillcolor=(124, 116, 104))
        base_ratio = float(output_size[1]) / output_size[0]
        self.random_resize_crop = IT.RandomResizedCrop(
            output_size, (0.8, 1),
            ratio=(base_ratio * 3. / 4., base_ratio * 4. / 3.),
            interpolation=Image.BICUBIC)
        self.to_tensor = TF.ToTensor()
        self.to_onehot = IT.ToOnehot(max_obj_n, shuffle=True)
        self.normalize = TF.Normalize((0.485, 0.456, 0.406),
                                      (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.img_list)

    def load_image_in_PIL(self, path, mode='RGB'):
        img = Image.open(path)
        img.load()  # Very important for loading large image
        return img.convert(mode)

    def sample_sequence(self, idx, gap=None):
        img_pil = self.load_image_in_PIL(self.img_list[idx], 'RGB')
        mask_pil = self.load_image_in_PIL(self.mask_list[idx], 'P')

        frames = []
        masks = []

        img_pil, mask_pil = self.pre_random_horizontal_flip(img_pil, mask_pil)

        for i in range(self.clip_n):
            img, mask = img_pil, mask_pil

            if i > 0:
                img, mask = self.random_horizontal_flip(img, mask)
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            img, mask = self.random_resize_crop(img, mask)

            mask = np.array(mask, np.uint8)

            if i == 0:
                mask, obj_list = self.to_onehot(mask)
                obj_num = len(obj_list)
            else:
                mask, _ = self.to_onehot(mask, obj_list)

            mask = torch.argmax(mask, dim=0, keepdim=True)

            frames.append(self.normalize(self.to_tensor(img)))
            masks.append(mask)

        sample = {
            'ref_img': frames[0],
            'prev_img': frames[1],
            'curr_img': frames[2:],
            'ref_label': masks[0],
            'prev_label': masks[1],
            'curr_label': masks[2:]
        }
        sample['meta'] = {
            'seq_name': self.img_list[idx],
            'frame_num': 1,
            'obj_num': obj_num
        }

        return sample, None

    def __getitem__(self, idx):
        sample1, _ = self.sample_sequence(idx)

        if self.dynamic_merge and (sample1['meta']['obj_num'] == 0
                                   or random.random() < self.merge_prob):
            rand_idx = np.random.randint(len(self.img_list))
            while (rand_idx == idx):
                rand_idx = np.random.randint(len(self.img_list))

            sample2, _ = self.sample_sequence(rand_idx)

            sample = self.merge_sample(sample1, sample2)
        else:
            sample = sample1

        return sample

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n)


class VOSTrain(Dataset):
    def __init__(self,
                 image_root,
                 label_root,
                 imglistdic,
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 merge_prob=0.3,
                 max_obj_n=10,
                 ignore_thresh=1.0,
                 ignore_in_merge=False):
        self.image_root = image_root
        self.label_root = label_root
        self.rand_gap = rand_gap
        self.seq_len = seq_len
        self.rand_reverse = rand_reverse
        self.repeat_time = repeat_time
        self.transform = transform
        self.dynamic_merge = dynamic_merge
        self.merge_prob = merge_prob
        self.enable_prev_frame = enable_prev_frame
        self.max_obj_n = max_obj_n
        self.rgb = rgb
        self.imglistdic = imglistdic
        self.seqs = list(self.imglistdic.keys())
        # maximum allowed fraction of ignored pixels to objects pixels for a frame to be used as a reference frame during training (used to avoid initilizaign the model with very noisy frames).
        self.ignore_thresh = ignore_thresh
        # if true, use the union of ignore regions of the two samples when merging them
        self.ignore_in_merge = ignore_in_merge
        print('Video Num: {} X {}'.format(len(self.seqs), self.repeat_time))

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def reverse_seq(self, imagelist, lablist):
        if np.random.randint(2) == 1:
            imagelist = imagelist[::-1]
            lablist = lablist[::-1]
        return imagelist, lablist

    def reload_images(self):
        for seq_name in self.imglistdic.keys():
            images = list(
                np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(self.label_root, seq_name))))
            self.imglistdic[seq_name] = (images, labels)

    def get_ref_index(self,
                      seqname,
                      lablist,
                      objs,
                      min_fg_pixels=200,
                      max_try=5):
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(len(lablist))
            if ref_index in bad_indices:
                continue
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            ref_objs = list(np.unique(ref_label))
            is_consistent = True
            for obj in ref_objs:
                if obj == 0:
                    continue
                if obj not in objs:
                    is_consistent = False
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels and is_consistent:
                break
            bad_indices.append(ref_index)
        return ref_index

    def get_ref_index_v2(self,
                         seqname,
                         lablist,
                         min_fg_pixels=200,
                         max_try=40,
                         total_gap=0,
                         ignore_thresh=0.2):
        search_range = len(lablist) - total_gap
        if search_range <= 1:
            return 0
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(search_range)
            if ref_index in bad_indices:
                continue
            frame_name = lablist[ref_index].split('.')[0] + '.jpg'
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index]))
            ref_label = np.array(ref_label, dtype=np.uint8)
            xs_ignore, ys_ignore = np.nonzero(ref_label == 255)
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels and (len(xs_ignore) / len(xs)) <= ignore_thresh:
                break
            bad_indices.append(ref_index)
        return ref_index

    def sample_gaps(self, seq_len, max_gap=99, max_try=10):
        for _ in range(max_try):
            curr_gaps = []
            total_gap = 0
            for _ in range(seq_len):
                gap = int(np.random.randint(self.rand_gap) + 1)
                # gap = 10
                total_gap += gap
                curr_gaps.append(gap)
            if total_gap <= max_gap:
                break
        return curr_gaps, total_gap


    def get_curr_gaps(self, seq_len, max_gap=99, max_try=10, labels=None, images=None, start_ind=0):
        curr_gaps, total_gap = self.sample_gaps(seq_len, max_gap, max_try)
        valid = False
        if start_ind + total_gap < len(images):
            label_name = images[start_ind + total_gap].split('.')[0] + '.png'
            if label_name in labels:
                valid = True
        count = 0
        while not valid and count < max_try:
            curr_gaps, total_gap = self.sample_gaps(seq_len, max_gap, max_try)
            valid = False
            count += 1
            if start_ind + total_gap < len(images):
                label_name = images[start_ind + total_gap].split('.')[0] + '.png'
                if label_name in labels:
                    valid = True

        if count == max_try:
            curr_gaps = [1] * min(seq_len, (len(images) - start_ind))
            if len(curr_gaps) < seq_len:
                curr_gaps += [0] * (seq_len - len(curr_gaps))
            total_gap = len(images) - start_ind

        return curr_gaps, total_gap

    def get_prev_index(self, lablist, total_gap):
        search_range = len(lablist) - total_gap
        if search_range > 1:
            prev_index = np.random.randint(search_range)
        else:
            prev_index = 0
        return prev_index

    def check_index(self, total_len, index, allow_reflect=True):
        if total_len <= 1:
            return 0

        if index < 0:
            if allow_reflect:
                index = -index
                index = self.check_index(total_len, index, True)
            else:
                index = 0
        elif index >= total_len:
            if allow_reflect:
                index = 2 * (total_len - 1) - index
                index = self.check_index(total_len, index, True)
            else:
                index = total_len - 1

        return index

    def get_curr_indices(self, imglist, prev_index, gaps):
        total_len = len(imglist)
        curr_indices = []
        now_index = prev_index
        for gap in gaps:
            now_index += gap
            curr_indices.append(self.check_index(total_len, now_index))
        return curr_indices

    def get_image_label(self, seqname, imagelist, lablist, index, is_ref=False):
        if is_ref:
            frame_name = lablist[index].split('.')[0] 
        else:
            frame_name = imagelist[index].split('.')[0]

        image = cv2.imread(
            os.path.join(self.image_root, seqname, frame_name + '.jpg'))
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        label_name = frame_name + '.png'
        if label_name in lablist:
            label = Image.open(
                os.path.join(self.label_root, seqname, label_name))
            label = np.array(label, dtype=np.uint8)
        else:
            label = None

        return image, label

    def sample_sequence(self, idx, dense_seq=None):
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]
        imagelist, lablist = self.imglistdic[seqname]
        frame_num = len(imagelist)
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 5
        try_step = 0
        while (is_consistent is False and try_step < max_try):
            try_step += 1

            # generate random gaps
            # curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1, labels=lablist, images=imagelist)

            if self.enable_prev_frame:  # prev frame is randomly sampled
                # get prev frame
                prev_index = self.get_prev_index(lablist, total_gap)
                prev_image, prev_label = self.get_image_label(
                    seqname, imagelist, lablist, prev_index)
                prev_objs = list(np.unique(prev_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, prev_index,
                                                     curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index)
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(prev_objs + curr_objs))

                start_index = prev_index
                end_index = max(curr_indices)
                # get ref frame
                _try_step = 0
                ref_index = self.get_ref_index_v2(seqname, lablist)
                while (ref_index > start_index and ref_index <= end_index
                       and _try_step < max_try):
                    _try_step += 1
                    ref_index = self.get_ref_index_v2(seqname, lablist)
                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index)
                ref_objs = list(np.unique(ref_label))
            else:  # prev frame is next to ref frame
                if dense_seq is None:
                    dense_seq = False
                # get ref frame
                ref_index = self.get_ref_index_v2(seqname, lablist, ignore_thresh=self.ignore_thresh, total_gap=self.seq_len)
                # frame_name = lablist[ref_index].split('.')[0] 
                # adjusted_index = imagelist.index(frame_name + '.jpg')
                adjusted_index = ref_index
                curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1, labels=lablist, images=imagelist, start_ind=adjusted_index)
                # if dense_seq:
                #     adjusted_index = imagelist.index(frame_name + '.jpg')
                #     if 'frames_eval' in self.image_root:
                #         self.rand_gap = 9
                #     curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1, labels=lablist, images=imagelist, start_ind=adjusted_index)
                # else: #to sample a fully-labeled sequence we sample indices from the labed frames only and then adjust them to the full range of frames  
                #     self.rand_gap = 3
                #     curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1, labels=lablist, images=lablist, start_ind=ref_index)
                #     adjusted_index = imagelist.index(frame_name + '.jpg')
                #     adjusted_curr_gaps = []
                #     for gap in curr_gaps:
                #         frame_name = lablist[ref_index + gap].split('.')[0] 
                #         adjusted_curr_gaps.append(imagelist.index(frame_name + '.jpg') - adjusted_index)
                #     curr_gaps = adjusted_curr_gaps

                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index, is_ref=True)
                ref_objs = list(np.unique(ref_label))

                # get curr frames
                curr_indices = self.get_curr_indices(imagelist, adjusted_index,
                                                     curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                labeled_frames = [1]
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index)
                    if curr_label is not None:
                        c_objs = list(np.unique(curr_label))
                        labeled_frames.append(1)
                    else:
                        curr_label = np.full_like(ref_label, 255)
                        c_objs = []
                        labeled_frames.append(0)
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(curr_objs))
                prev_image, prev_label = curr_images[0], curr_labels[0]
                curr_images, curr_labels = curr_images[1:], curr_labels[1:]

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_ids = list(np.sort(ref_objs))
        if 255 not in obj_ids:
            obj_num = obj_ids[-1]
        else:
            obj_num = obj_ids[-2]

        sample = {
            'ref_img': ref_image,
            'prev_img': prev_image,
            'curr_img': curr_images,
            'ref_label': ref_label,
            'prev_label': prev_label,
            'curr_label': curr_labels
        }
        sample['meta'] = {
            'seq_name': seqname,
            'frame_num': frame_num,
            'obj_num': obj_num,
            'dense_seq': dense_seq
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        sample1 = self.sample_sequence(idx)

        if self.dynamic_merge and not sample1['meta']['dense_seq'] and (sample1['meta']['obj_num'] == 0
                                   or random.random() < self.merge_prob):
            rand_idx = np.random.randint(len(self.seqs))
            while (rand_idx == (idx % len(self.seqs))):
                rand_idx = np.random.randint(len(self.seqs))

            sample2 = self.sample_sequence(rand_idx, False)

            sample = self.merge_sample(sample1, sample2)
        else:
            sample = sample1

        return sample

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n, self.ignore_in_merge)


class DAVIS2017_Train(VOSTrain):
    def __init__(self,
                 split=['train'],
                 root='./DAVIS',
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 full_resolution=True,
                 year=2017,
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3):
        if full_resolution:
            resolution = 'Full-Resolution'
            if not os.path.exists(os.path.join(root, 'JPEGImages',
                                               resolution)):
                print('No Full-Resolution, use 480p instead.')
                resolution = '480p'
        else:
            resolution = '480p'
        image_root = os.path.join(root, 'JPEGImages') # Corrected: image_root is base, resolution subdir handled later
        label_root = os.path.join(root, 'Annotations') # Corrected: label_root is base
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets', str(year),
                                   spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        imglistdic = {}
        for seq_name in seq_names:
            # Append resolution subdir here for actual file listing
            images = list(
                np.sort(os.listdir(os.path.join(image_root, resolution, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(label_root, resolution, seq_name))))
            imglistdic[seq_name] = (images, labels)

        # Pass the image_root and label_root that include the resolution subdir to VOSTrain
        super(DAVIS2017_Train, self).__init__(os.path.join(image_root, resolution), # Pass resolved path
                                              os.path.join(label_root, resolution), # Pass resolved path
                                              imglistdic,
                                              transform,
                                              rgb,
                                              repeat_time,
                                              rand_gap,
                                              seq_len,
                                              rand_reverse,
                                              dynamic_merge,
                                              enable_prev_frame,
                                              merge_prob=merge_prob,
                                              max_obj_n=max_obj_n)

class VOST_Train(VOSTrain):
    def __init__(self,
                 split=['train'],
                 root='./VOSTv0',
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 full_resolution=True, # Not used in VOST, but keep for consistency
                 year=2017, # Not used in VOST, but keep for consistency
                 rand_gap=3,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3,
                 ignore_thresh=1.0,
                 ignore_in_merge=False):
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        # valid_root = os.path.join(root, 'ValidAnns') # Not used in current VOSTrain
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets',
                                   spt + '.txt')) as f:
                seqs_tmp = f.readlines()
            seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
            seq_names.extend(seqs_tmp)
        imglistdic = {}
        for seq_name in seq_names:
            images = list(
                np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(label_root, seq_name))))
            imglistdic[seq_name] = (images, labels)

        super(VOST_Train, self).__init__(image_root,
                                              label_root,
                                              imglistdic,
                                              transform,
                                              rgb,
                                              repeat_time,
                                              rand_gap,
                                              seq_len,
                                              rand_reverse,
                                              dynamic_merge,
                                              enable_prev_frame,
                                              merge_prob=merge_prob,
                                              max_obj_n=max_obj_n,
                                              ignore_thresh=ignore_thresh,
                                              ignore_in_merge=ignore_in_merge)

class VISOR_Train(VOSTrain):
    def __init__(self,
                 split=['train'],
                 root='./VISOR',
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 full_resolution=True, # Not used directly in VISOR path construction by VOSTrain
                 year=2017, # Not used by VISOR
                 rand_gap=1, # Default for VISOR
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3,
                 ignore_thresh=1.0):
        image_root = os.path.join(root, 'JPEGImages') # VISOR specific structure
        label_root = os.path.join(root, 'Annotations') # VISOR specific structure
        seq_names = []
        # VISOR has seq_info.json instead of ImageSets txt files usually
        # This part needs to align with how VISOR sequences are listed.
        # Assuming a similar ImageSets/train.txt for now, or this needs custom logic.
        visor_set_file = os.path.join(root, 'ImageSets', split[0] + '.txt')
        if not os.path.exists(visor_set_file):
             print(f"Warning: VISOR image set file not found: {visor_set_file}. Dataset may be empty.")
             seqs_tmp = []
        else:
            with open(visor_set_file) as f:
                seqs_tmp = f.readlines()
        seqs_tmp = list(map(lambda elem: elem.strip(), seqs_tmp))
        seq_names.extend(seqs_tmp)

        imglistdic = {}
        for seq_name in seq_names:
            images = list(
                np.sort(os.listdir(os.path.join(image_root, seq_name))))
            labels = list(
                np.sort(os.listdir(os.path.join(label_root, seq_name))))
            imglistdic[seq_name] = (images, labels)

        super(VISOR_Train, self).__init__(image_root,
                                              label_root,
                                              imglistdic,
                                              transform,
                                              rgb,
                                              repeat_time,
                                              rand_gap,
                                              seq_len,
                                              rand_reverse,
                                              dynamic_merge,
                                              enable_prev_frame,
                                              merge_prob=merge_prob,
                                              max_obj_n=max_obj_n,
                                              ignore_thresh=ignore_thresh)


class YOUTUBEVOS_Train(VOSTrain):
    def __init__(self,
                 root='./datasets/YTB',
                 year=2019,
                 transform=None,
                 rgb=True,
                 rand_gap=3,
                 seq_len=3,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3):
        root = os.path.join(root, str(year), 'train')
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        self.seq_list_file = os.path.join(root, 'meta.json')
        self._check_preprocess()
        seq_names = list(self.ann_f.keys())

        imglistdic = {}
        for seq_name in seq_names:
            data = self.ann_f[seq_name]['objects']
            obj_names = list(data.keys())
            images = []
            labels = []
            # YTB specific: Iterate objects to gather all frames for a video
            all_frames_for_video = set()
            for obj_n in obj_names:
                all_frames_for_video.update(data[obj_n]["frames"])

            if not all_frames_for_video:
                print(f"No frames found for video: {seq_name}")
                continue

            images = sorted([f + '.jpg' for f in all_frames_for_video])
            labels = sorted([f + '.png' for f in all_frames_for_video]) # Assuming all jpgs have pngs

            # Verify all labels exist, or filter them. For YTB, typically they do.
            # This simplified version assumes they exist. A robust version would check.

            if len(images) < 2: # Or seq_len
                # print("Short video (after unique frame gathering): " + seq_name)
                continue
            imglistdic[seq_name] = (images, labels)

        super(YOUTUBEVOS_Train, self).__init__(image_root,
                                               label_root,
                                               imglistdic,
                                               transform,
                                               rgb,
                                               1, # repeat_time is 1 for YTB
                                               rand_gap,
                                               seq_len,
                                               rand_reverse,
                                               dynamic_merge,
                                               enable_prev_frame,
                                               merge_prob=merge_prob,
                                               max_obj_n=max_obj_n)

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            print('No such file: {}.'.format(self.seq_list_file))
            # Fallback: try to list sequences from directories if meta.json is missing
            if os.path.isdir(os.path.join(self.image_root)):
                 seqs_from_dirs = [d for d in os.listdir(self.image_root) if os.path.isdir(os.path.join(self.image_root, d))]
                 self.ann_f = {seq: {"objects": {}} for seq in seqs_from_dirs} # Dummy structure
                 print(f"Warning: meta.json not found. Using directory listing for sequences: {len(seqs_from_dirs)} found.")
                 return True # Allow proceeding with directory listing
            return False # Cannot proceed if no meta.json and no dirs
        else:
            self.ann_f = json.load(open(self.seq_list_file, 'r'))['videos']
            return True


class TEST(Dataset):
    def __init__(
        self,
        seq_len=3,
        obj_num=3,
        transform=None,
    ):
        self.seq_len = seq_len
        self.obj_num = obj_num
        self.transform = transform

    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        img = np.zeros((800, 800, 3)).astype(np.float32)
        label = np.ones((800, 800)).astype(np.uint8)
        sample = {
            'ref_img': img,
            'prev_img': img,
            'curr_img': [img] * (self.seq_len - 2),
            'ref_label': label,
            'prev_label': label,
            'curr_label': [label] * (self.seq_len - 2)
        }
        sample['meta'] = {
            'seq_name': 'test',
            'frame_num': 100,
            'obj_num': self.obj_num
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def polygon_to_mask(shapes, height, width):
    """
    Convert polygon annotations to a mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for i, shape in enumerate(shapes):
        points = np.array(shape['points'], dtype=np.int32)
        # Assuming a single class for now, use object index + 1 as fill value
        # Add 1 because 0 is typically background
        cv2.fillPoly(mask, [points], color=(i + 1))
    return mask


import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset # Ensure this is the direct parent

# polygon_to_mask should be defined globally in the file, e.g.:
# def polygon_to_mask(shapes, height, width):
#     mask = np.zeros((height, width), dtype=np.uint8)
#     for i, shape in enumerate(shapes):
#         points = np.array(shape['points'], dtype=np.int32)
#         cv2.fillPoly(mask, [points], color=(i + 1)) # Object IDs start from 1
#     return mask

class ExtractedFramesTrain(Dataset):
    def __init__(self,
                 image_root="extracted_frames/",
                 transform=None,
                 rgb=True,
                 seq_len=5,
                 max_obj_n=10, # Kept for signature compatibility
                 repeat_time=1,
                 ignore_thresh=1.0): # Kept for signature compatibility

        super(ExtractedFramesTrain, self).__init__()

        self.image_root = image_root
        self.transform = transform
        self.rgb = rgb
        self.seq_len = seq_len
        self.repeat_time = repeat_time

        # Store these for compatibility, though not actively used in this simplified loader's core logic
        self.max_obj_n = max_obj_n
        self.ignore_thresh = ignore_thresh

        if self.seq_len < 1:
            raise ValueError("ExtractedFramesTrain: seq_len must be at least 1.")
        # The structure ref/prev/curr implies seq_len >= 2 for prev, and seq_len >= 3 for curr[0]
        # This warning can be adjusted based on how Trainer uses the sample.
        if self.seq_len < 2 : # Strictly, prev_img might not be valid if seq_len is 1
             print(f"Warning: ExtractedFramesTrain initialized with seq_len={self.seq_len}. "
                   f"The sample structure might not fully populate 'prev_img' or 'curr_img'.")

        self.all_image_files = []
        self.img_to_json_map = {}
        self.valid_sequence_start_indices = []

        print(f"ExtractedFramesTrain: Scanning for images in: {os.path.abspath(self.image_root)}")

        candidate_image_files = []
        if not os.path.isdir(self.image_root):
            print(f"Warning: ExtractedFramesTrain - Image root directory {self.image_root} does not exist.")
        else:
            for dirpath, _, filenames in os.walk(self.image_root):
                for filename in sorted(filenames): # Process in sorted order
                    if filename.lower().endswith('.jpg'):
                        candidate_image_files.append(os.path.join(dirpath, filename))

        for img_path in candidate_image_files:
            json_path = os.path.splitext(img_path)[0] + '.json'
            if os.path.exists(json_path):
                self.all_image_files.append(img_path)
                self.img_to_json_map[img_path] = json_path
            # else:
                # print(f"Debug: JSON for {img_path} not found. Skipping.")

        if not self.all_image_files:
            print(f"Warning: ExtractedFramesTrain - No valid image-JSON pairs found in {self.image_root}. Dataset will be empty.")
            # self.valid_sequence_start_indices will remain empty, len will be 0.
            return

        print(f"ExtractedFramesTrain: Found {len(self.all_image_files)} images with corresponding JSON files.")

        if len(self.all_image_files) >= self.seq_len:
            for i in range(len(self.all_image_files) - self.seq_len + 1):
                first_frame_dir = os.path.dirname(self.all_image_files[i])
                is_coherent_sequence = True
                for k in range(1, self.seq_len):
                    if os.path.dirname(self.all_image_files[i + k]) != first_frame_dir:
                        is_coherent_sequence = False
                        break
                if is_coherent_sequence:
                    self.valid_sequence_start_indices.append(i)

        if not self.valid_sequence_start_indices:
            print(f"Warning: ExtractedFramesTrain - No valid sequences of length {self.seq_len} could be formed from {len(self.all_image_files)} images.")
        else:
            print(f"ExtractedFramesTrain: Initialized. Found {len(self.valid_sequence_start_indices)} valid starting points for sequences of length {self.seq_len}.")

    def __len__(self):
        if not self.valid_sequence_start_indices:
            return 0
        return len(self.valid_sequence_start_indices) * self.repeat_time

    def __getitem__(self, idx):
        if not self.valid_sequence_start_indices:
            raise IndexError("ExtractedFramesTrain: No valid sequences to sample from.")

        # Adjust index based on repeat_time
        actual_sample_idx = idx % len(self.valid_sequence_start_indices)
        start_file_list_index = self.valid_sequence_start_indices[actual_sample_idx]

        loaded_images = []
        loaded_labels = []
        frame_meta_names = []

        # Determine the directory of the first frame in the sequence for seq_name metadata
        current_sequence_dir = os.path.dirname(self.all_image_files[start_file_list_index])

        for i in range(self.seq_len):
            current_frame_global_idx = start_file_list_index + i
            img_path = self.all_image_files[current_frame_global_idx]
            json_path = self.img_to_json_map[img_path] # Assumes all files in a valid sequence have JSONs

            frame_meta_names.append(os.path.basename(img_path))

            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"ExtractedFramesTrain: Image not found during sequence loading: {img_path}")

            img_h, img_w, _ = image.shape
            image_np = np.array(image, dtype=np.float32)

            if self.rgb:
                image_np = image_np[:, :, [2, 1, 0]]

            mask_h, mask_w = img_h, img_w
            label_np = np.zeros((mask_h, mask_w), dtype=np.uint8) # Default empty mask

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        annot_data = json.load(f)
                        shapes = annot_data.get('shapes', [])
                        json_h = annot_data.get('imageHeight', img_h)
                        json_w = annot_data.get('imageWidth', img_w)
                        if shapes:
                            label_np = polygon_to_mask(shapes, json_h, json_w)
                    except json.JSONDecodeError as e:
                        print(f"ExtractedFramesTrain: Error decoding JSON {json_path}: {e}. Using empty mask.")
            # No else needed here as we pre-filtered all_image_files to only include those with JSONs

            loaded_images.append(image_np)
            loaded_labels.append(label_np)

        sample = {}
        sample['ref_img'] = loaded_images[0]
        sample['ref_label'] = loaded_labels[0]

        if self.seq_len >= 2:
            sample['prev_img'] = loaded_images[1]
            sample['prev_label'] = loaded_labels[1]
            # curr_img/label should be a list of remaining frames
            sample['curr_img'] = loaded_images[2:]
            sample['curr_label'] = loaded_labels[2:]
        else: # seq_len == 1
            sample['prev_img'] = loaded_images[0].copy() # Use ref as prev
            sample['prev_label'] = loaded_labels[0].copy()
            sample['curr_img'] = [] # Empty list
            sample['curr_label'] = [] # Empty list

        obj_ids_ref = list(np.unique(loaded_labels[0]))
        obj_num = 0
        valid_obj_ids = [oid for oid in obj_ids_ref if oid != 0 and oid != 255] # Assuming 255 is an ignore label
        if valid_obj_ids:
            obj_num = max(valid_obj_ids)

        # Determine seq_name for meta
        # If image_root is "extracted_frames" and frames are directly under it, seq_name might be "extracted_frames"
        # If frames are under "extracted_frames/video1", "extracted_frames/video2", then seq_name is "video1", "video2"
        # current_sequence_dir gives the full path to the directory of the first frame.
        if self.image_root == current_sequence_dir or os.path.abspath(self.image_root) == os.path.abspath(current_sequence_dir) :
             # Frames are directly in image_root, use a generic name or first frame name
            seq_name_for_meta = os.path.splitext(frame_meta_names[0])[0] # Use first frame name as proxy for seq name
        else:
            seq_name_for_meta = os.path.basename(current_sequence_dir)


        sample['meta'] = {
            'seq_name': seq_name_for_meta,
            'frame_num': len(self.all_image_files), # Total frames in this specific "video" or sub-folder
            'obj_num': obj_num,
            'dense_seq': True, # All frames are "real" for this loader
            'frame_names': frame_meta_names # List of frame filenames in the current sequence
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
