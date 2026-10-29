"""
Module: video_datasets.py

This module provides classes and functions for loading and processing video datasets,
splitting them into training, validation, and test sets, and preparing data for video
classification models. It includes a custom PyTorch Dataset for videos stored as directories
of frame images, functions to load the dataset from a directory structure, split the dataset
using stratified sampling, and custom collate functions for handling variable-length video sequences.
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from PIL import Image

import os
import glob
import random
import re
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def _group_id(video_path):
    """Return the UCF group number from a clip name, e.g. g08 from v_Punch_g08_c01."""
    name = os.path.basename(str(video_path).rstrip("/"))
    match = re.search(r"_g(\d+)_", name)
    if match is None:
        return None
    return int(match.group(1))


def _frame_sort_key(frame_path):
    """Sort key so frame2.jpg comes before frame10.jpg (not lexicographic)."""
    digits = re.findall(r"\d+", os.path.basename(frame_path))
    if digits:
        return [int(n) for n in digits]
    return [os.path.basename(frame_path)]


def _select_frame_paths(fr_paths, n_frames):
    """Pick n_frames paths uniformly across the clip, not just the first JPEGs."""
    if len(fr_paths) <= n_frames:
        return fr_paths
    idxs = np.linspace(0, len(fr_paths) - 1, n_frames, dtype=np.int32)
    return [fr_paths[int(i)] for i in idxs]


class VideoDataset(Dataset):
    """
    PyTorch Dataset class for loading video data from directories of frame images.

    Each video is represented as a directory containing JPEG images of its frames.
    The dataset is provided as a dictionary mapping each video directory path to its label.

    Args:
        vid_dataset (dict): Dictionary where keys are video directory paths and values are integer labels.
        fr_per_vid (int): Number of frames per video to load.
        transforms (callable, optional): A function/transform to apply to each frame image (e.g., resizing, normalization).
    """
    def __init__(self, vid_dataset, fr_per_vid, transforms=None):
        self.dataset = vid_dataset
        self.fpv = fr_per_vid
        self.img_transforms = transforms

    def __len__(self):
        """Return the number of video samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Load frames from the video directory corresponding to the given index, apply transforms,
        and return the stacked tensor of frames along with its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (frames_tensor, label) where frames_tensor is a tensor of shape (T, C, H, W)
                   with T being the number of frames (up to fr_per_vid) and label is an integer.
        """
        # Sort by frame number so order is time, not filesystem glob order.
        # (unsorted glob can yield frame10 before frame2; [:fpv] would then drop later frames.)
        fr_paths = glob.glob(os.path.join(self.dataset[idx][0], "*.jpg"))
        fr_paths = sorted(fr_paths, key=_frame_sort_key)
        fr_paths = _select_frame_paths(fr_paths, self.fpv)

        # Open images using PIL
        fr_imgs = [Image.open(fr_path) for fr_path in fr_paths]

        # Get the label associated with the video
        fr_label = self.dataset[idx][1]

        # Same random seed on every frame so flip/crop stay temporally consistent.
        if self.img_transforms:
            seed = random.randint(0, 2**31 - 1)
            fr_imgs_trans = []
            for fr_img in fr_imgs:
                random.seed(seed)
                torch.manual_seed(seed)
                fr_imgs_trans.append(self.img_transforms(fr_img))
        else:
            fr_imgs_trans = fr_imgs

        # Stack transformed images into a tensor if available
        if len(fr_imgs_trans) > 0:
            fr_imgs_trans = torch.stack(fr_imgs_trans)

        return fr_imgs_trans, fr_label


def load_dataset(frame_dir):
    """
    Load the full video dataset from the specified directory.

    Each subdirectory in frame_dir is assumed to correspond to a video category.
    The function builds a dictionary where keys are paths to video directories and
    values are integer labels corresponding to each category.

    Args:
        frame_dir (str): Path to the directory containing subdirectories for each video category.

    Returns:
        tuple: (vid_dataset, label_dict)
            - vid_dataset (dict): Dictionary mapping video directory paths to integer labels.
            - label_dict (dict): Dictionary mapping video category names to integer labels.
    """
    class_names = [
        name for name in sorted(os.listdir(frame_dir))
        if not name.startswith(".") and os.path.isdir(os.path.join(frame_dir, name))
    ]
    label_dict = {vid_cat: idx for idx, vid_cat in enumerate(class_names)}
    vid_dataset = {}
    print('Loading video dataset....')
    for vid_cat in tqdm(class_names):
        vid_cat_path = os.path.join(frame_dir, vid_cat)
        for vid in os.listdir(vid_cat_path):
            if vid.startswith("."):
                continue
            vid_path = os.path.join(vid_cat_path, vid)
            if not os.path.isdir(vid_path):
                continue
            vid_dataset[vid_path] = label_dict[vid_cat]
    return vid_dataset, label_dict


def dataset_split(vid_dataset, tr_ratio, ts_ratio, seed=0):
    """
    Split the dataset into training, validation, and test sets using stratified sampling.

    This function uses StratifiedShuffleSplit to ensure that each split has a representative
    distribution of classes.

    Args:
        vid_dataset (dict): Dictionary mapping video paths to labels.
        tr_ratio (float): Proportion of the data to use for training.
        ts_ratio (float): Proportion of the data to use for testing.
        seed (int, optional): Random seed for reproducibility. Default is 0.

    Returns:
        tuple: (tr_dataset, val_dataset, ts_dataset)
            - tr_dataset (list): List of (video_path, label) tuples for the training set.
            - val_dataset (list): List of (video_path, label) tuples for the validation set.
            - ts_dataset (list): List of (video_path, label) tuples for the test set.
    """
    samples = list(vid_dataset.items()) if isinstance(vid_dataset, dict) else list(vid_dataset)

    groups = defaultdict(list)
    for path, label in samples:
        gid = _group_id(path)
        # If a name has no group token, treat that clip as its own group.
        group_key = (int(label), gid if gid is not None else path)
        groups[group_key].append((path, int(label)))

    group_keys = np.array(list(groups.keys()), dtype=object)
    group_labels = np.array([key[0] for key in group_keys])


    ts_spliter = StratifiedShuffleSplit(n_splits=1, test_size=ts_ratio, random_state=seed)
    tr_val_idx, ts_idx = next(ts_spliter.split(group_keys, group_labels))

    val_ratio = 1.0 - tr_ratio - ts_ratio
    val_wt = val_ratio / (tr_ratio + val_ratio)
    val_spliter = StratifiedShuffleSplit(n_splits=1, test_size=val_wt, random_state=seed)
    tr_idx, val_idx = next(
        val_spliter.split(group_keys[tr_val_idx], group_labels[tr_val_idx])
    )

    def _expand(keys):
        clips = []
        for key in keys:
            clips.extend(groups[tuple(key)])
        return clips

    tr_dataset = _expand(group_keys[tr_val_idx][tr_idx])
    val_dataset = _expand(group_keys[tr_val_idx][val_idx])
    ts_dataset = _expand(group_keys[ts_idx])
    return tr_dataset, val_dataset, ts_dataset


def collate_fn_r3d_18(batch):
    """
    Collate function for 3D CNN models (e.g., R3D-18).

    Assumes each sample in the batch is a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). This function filters out any samples
    with no frames, stacks the video frame tensors, transposes the tensor dimensions as needed,
    and stacks the labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (imgs_tensor, labels_tensor)
            - imgs_tensor (Tensor): Stacked video frames tensor with shape adjusted for R3D-18.
            - labels_tensor (Tensor): Tensor of labels.
    """
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def collate_fn_rnn(batch):
    """
    Collate function for RNN-based models.

    Handles variable-length video sequences by padding them to the length of the longest sequence
    in the batch. Each sample in the batch is expected to be a tuple (video_frames, label),
    where video_frames is a tensor of shape (T, C, H, W). The function returns a padded tensor
    of video frames with shape (batch_size, max_T, C, H, W) and a tensor of labels.

    Args:
        batch (list): List of samples, each as (video_frames, label).

    Returns:
        tuple: (padded_imgs, labels_tensor)
            - padded_imgs (Tensor): Padded tensor of video frames.
            - labels_tensor (Tensor): Tensor of labels.
    """
    # Unzip the batch into image tensors and labels
    imgs_batch, label_batch = list(zip(*batch))

    # Filter out any samples that have no frames
    valid_samples = [(imgs, label) for imgs, label in zip(imgs_batch, label_batch) if len(imgs) > 0]
    if not valid_samples:
        return None, None
    imgs_batch, label_batch = zip(*valid_samples)

    # Pad the video frame tensors along the time dimension (T)
    # Resulting shape: (batch_size, max_T, C, H, W)
    padded_imgs = pad_sequence(imgs_batch, batch_first=True)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(label_batch)

    return padded_imgs, labels_tensor
