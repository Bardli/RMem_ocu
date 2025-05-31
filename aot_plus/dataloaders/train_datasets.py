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
from torch.utils.data import Dataset
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
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        seq_names = []
        for spt in split:
            with open(os.path.join(root, 'ImageSets', str(year),
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

        super(DAVIS2017_Train, self).__init__(image_root,
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
                                              max_obj_n=max_obj_n)

class VOST_Train(VOSTrain):
    def __init__(self,
                 split=['train'],
                 root='./VOSTv0',
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
                 merge_prob=0.3,
                 ignore_thresh=1.0,
                 ignore_in_merge=False):
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
        valid_root = os.path.join(root, 'ValidAnns')
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
                 full_resolution=True,
                 year=2017,
                 rand_gap=1,
                 seq_len=5,
                 rand_reverse=True,
                 dynamic_merge=True,
                 enable_prev_frame=False,
                 max_obj_n=10,
                 merge_prob=0.3,
                 ignore_thresh=1.0):
        image_root = os.path.join(root, 'JPEGImages')
        label_root = os.path.join(root, 'Annotations')
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
            for obj_n in obj_names:
                if len(data[obj_n]["frames"]) < 2:
                    print("Short object: " + seq_name + '-' + obj_n)
                    continue
                images += list(
                    map(lambda x: x + '.jpg', list(data[obj_n]["frames"])))
                labels += list(
                    map(lambda x: x + '.png', list(data[obj_n]["frames"])))
            images = np.sort(np.unique(images))
            labels = np.sort(np.unique(labels))
            if len(images) < 2:
                print("Short video: " + seq_name)
                continue
            imglistdic[seq_name] = (images, labels)

        super(YOUTUBEVOS_Train, self).__init__(image_root,
                                               label_root,
                                               imglistdic,
                                               transform,
                                               rgb,
                                               1,
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
            return False
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


class ExtractedFramesTrain(VOSTrain):
    def __init__(self,
                 image_root='extracted_frames/', # Default path
                 label_root='extracted_frames/', # Default path for JSONs
                 transform=None,
                 rgb=True,
                 repeat_time=1,
                 seq_len=5, # Default sequence length
                 max_obj_n=10,
                 # These are from VOSTrain, pass them along or set sensible defaults
                 rand_gap=1, # Set to 1 for mostly contiguous sequences from found files
                 rand_reverse=False, # Usually False for extracted video frames
                 dynamic_merge=False, # Disable merging for this loader
                 enable_prev_frame=False, # Our logic handles this directly
                 merge_prob=0.0,
                 ignore_thresh=1.0
                 ):

        # Initialize basic attributes from VOSTrain that we might not heavily use but superclass expects
        # We are largely bypassing VOSTrain's list-based indexing for our file-based one.
        # So, imglistdic passed to super can be minimal.
        # The critical part is that self.transform is set.
        super(ExtractedFramesTrain, self).__init__(
            image_root=image_root,
            label_root=label_root,
            imglistdic={}, # Pass empty, as we manage file lists directly
            transform=transform,
            rgb=rgb,
            repeat_time=repeat_time,
            rand_gap=rand_gap,
            seq_len=seq_len, # Pass our seq_len
            rand_reverse=rand_reverse,
            dynamic_merge=dynamic_merge,
            enable_prev_frame=enable_prev_frame,
            merge_prob=merge_prob,
            max_obj_n=max_obj_n,
            ignore_thresh=ignore_thresh
        )

        self.image_root = image_root
        self.label_root = label_root # Where JSONs are, typically same as image_root
        self.seq_len = seq_len
        self.rgb = rgb
        self.transform = transform
        self.max_obj_n = max_obj_n
        self.repeat_time = repeat_time # For __len__

        self.all_image_files = []
        self.img_to_json_map = {}

        # Scan for .jpg files and their .json counterparts
        if not os.path.isdir(self.image_root):
            print(f"Warning: Image root directory {self.image_root} does not exist.")
            self.valid_sequence_start_indices = []
            return

        for dirpath, _, filenames in os.walk(self.image_root):
            for filename in sorted(filenames): # Sort to maintain order
                if filename.lower().endswith('.jpg'):
                    img_full_path = os.path.join(dirpath, filename)
                    json_filename = os.path.splitext(filename)[0] + '.json'
                    json_full_path = os.path.join(dirpath, json_filename)

                    if os.path.exists(json_full_path):
                        self.all_image_files.append(img_full_path)
                        self.img_to_json_map[img_full_path] = json_full_path
                    # else:
                        # print(f"Debug: JSON file for {img_full_path} not found. Skipping.")

        # self.all_image_files should already be sorted if filenames from os.walk were sorted
        # and processed in that order for a single directory. If multiple subdirs, this sort is important.
        # For simplicity, current os.walk processes directory by directory. If frames from one sequence
        # are in one directory, sorting filenames is enough. If they are spread, more complex logic needed.
        # Assuming frames for a sequence are in the same directory and sorted filenames are enough.
        # self.all_image_files.sort() # Ensure overall sort if multiple subdirs were walked out of order

        self.valid_sequence_start_indices = []
        if self.seq_len > 0 and len(self.all_image_files) >= self.seq_len:
            for i in range(len(self.all_image_files) - self.seq_len + 1):
                # Basic check: are files from the same directory? (implies same sequence)
                # This check assumes a flat directory structure for frames of a single video,
                # or that self.all_image_files correctly groups by sequence.
                # For now, we assume all_image_files are globally sorted, and any slice of seq_len is a candidate.
                # A more robust check would ensure all files in a candidate sequence belong to the same video
                # (e.g. share the same parent directory if image_root has subdirs per video)
                # For this refactor, we assume files are sorted and any segment is fine.
                # This might need refinement if dataset has multiple unrelated sequences mixed.
                # Let's assume for now that all_image_files contains frames from ONE sequence or
                # that the sorting naturally groups them correctly for slicing.

                # Example check (can be enhanced):
                # first_frame_dir = os.path.dirname(self.all_image_files[i])
                # last_frame_dir = os.path.dirname(self.all_image_files[i + self.seq_len - 1])
                # if first_frame_dir == last_frame_dir:
                #    self.valid_sequence_start_indices.append(i)
                # else:
                #    print(f"Debug: Sequence starting at {self.all_image_files[i]} spans directories. Not adding.")
                self.valid_sequence_start_indices.append(i) # Simplified: any slice is a candidate


        if not self.valid_sequence_start_indices:
            print(f"Warning: No valid sequences of length {self.seq_len} could be formed from {len(self.all_image_files)} images.")

        print(f'ExtractedFramesTrain: {len(self.all_image_files)} images found. {len(self.valid_sequence_start_indices)} potential sequences of length {self.seq_len}. Repeat time: {self.repeat_time}.')

    def __len__(self):
        return len(self.valid_sequence_start_indices) * self.repeat_time

    def sample_sequence(self, idx, dense_seq=None): # dense_seq might not be used here
        if not self.valid_sequence_start_indices:
            raise IndexError("No valid sequences available in ExtractedFramesTrain.")

        actual_start_file_idx = self.valid_sequence_start_indices[idx % len(self.valid_sequence_start_indices)]

        loaded_images = []
        loaded_labels = []
        frame_meta_names = []

        # Determine sequence name (e.g., parent directory of the first frame)
        # This assumes image_root might contain subdirectories for different sequences.
        # If image_root is flat, then seq_name might just be image_root or a fixed string.
        first_frame_path = self.all_image_files[actual_start_file_idx]
        seq_name_candidate = os.path.basename(os.path.dirname(first_frame_path))
        if seq_name_candidate == os.path.basename(self.image_root.rstrip('/\\')): # if parent is image_root itself
             seq_name_candidate = os.path.basename(self.image_root) # use name of image_root

        meta_seq_name = seq_name_candidate if seq_name_candidate else "default_sequence"


        for i in range(self.seq_len):
            frame_path = self.all_image_files[actual_start_file_idx + i]
            json_path = self.img_to_json_map[frame_path]

            image = cv2.imread(frame_path)
            if image is None:
                raise FileNotFoundError(f"Image not found during sequence sampling: {frame_path}")
            image = np.array(image, dtype=np.float32)

            if self.rgb:
                image = image[:, :, [2, 1, 0]] # BGR to RGB

            height, width, _ = image.shape
            label = np.zeros((height, width), dtype=np.uint8) # Default empty mask

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        annot_data = json.load(f)
                        shapes = annot_data.get('shapes', [])
                        json_height = annot_data.get('imageHeight', height) # Use image's height if not in json
                        json_width = annot_data.get('imageWidth', width)   # Use image's width if not in json
                        if shapes:
                            label = polygon_to_mask(shapes, json_height, json_width)
                        # else: print(f"Debug: No shapes in {json_path}")
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON {json_path}: {e}. Using empty mask.")
            # else: print(f"Debug: JSON {json_path} not found. Using empty mask.")

            loaded_images.append(image)
            loaded_labels.append(label)
            frame_meta_names.append(os.path.basename(frame_path))

        if self.seq_len <= 0:
            raise ValueError("Sequence length must be at least 1.")
        # Based on problem description:
        # ref_img = loaded_images[0]
        # prev_img = loaded_images[1] (if seq_len >= 2, else copy of ref_img for seq_len==1)
        # curr_img = loaded_images[2:] (empty if seq_len < 3)

        ref_img = loaded_images[0]
        ref_label = loaded_labels[0]

        if self.seq_len == 1:
            prev_img = loaded_images[0] # Copy of ref
            prev_label = loaded_labels[0] # Copy of ref
            curr_img_list = []
            curr_label_list = []
        else: # seq_len >= 2
            prev_img = loaded_images[1]
            prev_label = loaded_labels[1]
            if self.seq_len == 2:
                curr_img_list = []
                curr_label_list = []
            else: # seq_len > 2
                curr_img_list = loaded_images[2:]
                curr_label_list = loaded_labels[2:]

        # Determine object number from the reference frame's mask
        obj_ids = list(np.unique(ref_label))
        obj_num = 0
        # Count unique values in the mask, excluding 0 (background) and 255 (ignore)
        valid_obj_ids = [oid for oid in obj_ids if oid != 0 and oid != 255]
        if valid_obj_ids:
            # obj_num can be max id or count of ids, depending on how it's used.
            # Using max ID is common if IDs are consecutive instance numbers.
            obj_num = max(valid_obj_ids)


        sample = {
            'ref_img': ref_img,
            'ref_label': ref_label,
            'prev_img': prev_img,
            'prev_label': prev_label,
            'curr_img': curr_img_list,
            'curr_label': curr_label_list
        }
        
        sample['meta'] = {
            'seq_name': meta_seq_name,
            'frame_num': self.seq_len, # Total frames in this sampled sequence
            'obj_num': obj_num,
            'dense_seq': True, # Assuming all frames in sequence are used
            'frame_names': frame_meta_names
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        # This directly calls sample_sequence, bypassing VOSTrain's __getitem__ logic
        # which might involve dynamic merging or other complexities not needed here.
        return self.sample_sequence(idx)
