# Copyright 2022 - VDIGPKU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import torch
import scipy
import scipy.ndimage
import scipy.stats as stats
import cv2 as cv
import torch.nn.functional as F

import numpy as np
from datasets import bbox_iou

def FORMULA(feats, feats_init, dims, scales, init_image_size, k_patches=100):
    """
    Implementation of FORMULA method.
    Inputs
        feats: the pixel/patche features of an image
        dims: dimension of the map from which the features are used
        scales: from image to map scale
        init_image_size: size of the image
        k_patches: number of k patches retrieved that are compared to the seed at seed expansion
    Outputs
        pred: box predictions
        A: binary affinity matrix
        scores: lowest degree scores for all patches
        seed: selected patch corresponding to an object
    """
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()  # [num_patch, num_patch]
    A_init = (feats_init @ feats_init.transpose(1, 2)).squeeze()

    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A)
    sorted_patches_init, scores_init = patch_scoring(A_init)

    # seed = torch.tensor([389]).cuda()
    # seed = 389
    # potentials = sorted_patches[:k_patches]
    # similars = potentials[A[seed, potentials] > 0.0]
    # M = torch.sum(A[similars, :], dim=0)

    # pred, _ = detect_box([M],
    #                      [seed],
    #                      dims,
    #                      scales=scales,
    #                      initial_im_size=init_image_size[1:])

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # return np.asarray(pred), A, scores, seed

    # s, m = visualize_attn(M, similars, seed, dims, scales)

    # Select the initial seed
    # seed = sorted_patches[0]
    seed = sorted_patches_init[0]

    # Seed expansion
    potentials = sorted_patches[:k_patches]  # Dk, INDICES of k patches

    similars = potentials[A[seed, potentials] > 0.0]  # S
    M = torch.sum(A[similars, :], dim=0)  # mask m
    # init_M = M

    runs = 4
    _seed = [seed]
    _M = [M]
    sims, masks = visualize_attn(M, similars, seed, dims, scales)

    for _ in range(runs):
        seed, wax = new_seed(M, _seed, dims)

        similars = torch.cat(
            (potentials[A[seed, potentials] > 0.0], wax)).long()
        M = torch.sum(A[similars, :], dim=0)

        # correl = M.reshape(dims[0], dims[1]).float()
        # labeled_array, _ = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

        # cc = labeled_array[seedy, seedx]
        # t, _ = np.where(labeled_array == cc)
        # if len(t) > 0.85 * dims[0] * dims[1]:
        #     _seed = [init_seed]
        #     _M = [init_M]
        #     break

        # if _ == 1:
        #     dis = visualize_attn(M, similars, seed, dims, scales)[1]
        #     dis = np.hstack((masks, m, dis))
        #     cv.imshow('...', dis)
        #     # cv.imwrite('./1.png', dis)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()
        #     exit(0)

        _seed.append(seed)
        _M.append(M)
        sims = np.hstack((sims, visualize_attn(M, similars, seed, dims,
                                               scales)[0]))
        masks = np.hstack(
            (masks, visualize_attn(M, similars, seed, dims, scales)[1]))

    # final_set = torch.tensor(_seed[1:])
    # M = torch.sum(A[final_set, :], dim=0)

    _seed.reverse()
    _M.reverse()

    sims = cv.resize(sims, dsize=None, fx=0.5, fy=0.5)
    masks = cv.resize(masks, dsize=None, fx=0.5, fy=0.5)

    # cv.imshow('similars', sims)
    # cv.imshow('masks', masks)

    # Box extraction
    pred, _ = detect_box(_M,
                         _seed,
                         dims,
                         scales=scales,
                         initial_im_size=init_image_size[1:])

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return np.asarray(pred), A, scores, _seed[-1]


def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


def new_seed(M, seeds, dims):
    """
    selecting a new seed as the center of the mask generated by FORMULA.
    """
    seed = seeds[-1]  # the latest seed

    correl = M.reshape(dims).float()

    # Compute connected components
    labeled_array, _ = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), dims)]
    Y, X = np.where(labeled_array == cc)  # row, col

    # Should not happen with FORMULA
    if cc == 0:
        # print('!' * 30)
        return seed, torch.tensor([]).cuda()
        # raise ValueError("The seed is in the background component.")

    res = []
    center_r, center_c = int(np.sum(Y) / len(Y) + 0.5), int(np.sum(X) / len(X) + 0.5)

    for r, c in zip(Y, X):
        d = np.sqrt((r - center_r)**2 + (c - center_c)**2)
        res.append([r, c, d])

    res = torch.tensor(res)
    res_sorted = res[res[:, -1].argsort(descending=True)]

    # t = int(np.sqrt(len(X)) * 0.5 / 2)
    t = 2
    sigma = 0.1
    new_seed_r = stats.truncnorm.rvs(-t / sigma,
                                     t / sigma,
                                     loc=center_r,
                                     scale=sigma,
                                     size=1)
    new_seed_c = stats.truncnorm.rvs(-t / sigma,
                                     t / sigma,
                                     loc=center_c,
                                     scale=sigma,
                                     size=1)
    # new_seed = int(new_seed_r[0]) * dims[1] + int(new_seed_c[0])
    new_seed = int(new_seed_r[0] + 0.5) * dims[1] + int(new_seed_c[0] + 0.5)
    # new_seed = center_r * dims[1] + center_c

    ratio = int(len(Y) * 0.12)
    wax = res_sorted[-ratio:, :2]
    wax = wax[:, 0] * dims[1] + wax[:, 1]

    return torch.tensor(new_seed).cuda(), torch.tensor(wax).cuda()


def detect_box(As, seeds, dims, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    A = As[0]
    seed = seeds[0]

    correl = A.reshape(dims).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(
        correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), dims)]

    # Should not happen with FORMULA
    i = 0
    while cc == 0 and i < len(As):
        # print('!' * 40)
        correl = As[i].reshape(dims).float()
        labeled_array, _ = scipy.ndimage.label(correl.cpu().numpy() > 0.0)
        cc = labeled_array[np.unravel_index(seeds[i].cpu().numpy(), dims)]
        i += 1
        # raise ValueError("The seed is in the background component.")

    # Find box
    mask = np.where(labeled_array == cc)

    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image size
    r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax  # scales: [16, 16]
    r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax

    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    # Check not out of image size (used when padding)
    if initial_im_size:
        pred[2] = min(pred[2], initial_im_size[1])
        pred[3] = min(pred[3], initial_im_size[0])

    # Coordinate predictions for the feature space
    # Axis different then in image space
    pred_feats = [ymin, xmin, ymax, xmax]

    return pred, pred_feats


def visualize_attn(M, similars, seed, dims, scales):
    """
    Visualization of the similars and the mask M.
    """
    w_featmap, h_featmap = dims

    sims = np.zeros((3, w_featmap * h_featmap))
    sims[:, similars.cpu()] = 255
    sims[:, seed] = [41 / 255, 37 / 255, 204 / 255]
    sims = sims.reshape((3, w_featmap, h_featmap))

    sims = F.interpolate(torch.from_numpy(sims).unsqueeze(0),
                         scale_factor=scales,
                         mode='nearest')[0].cpu().numpy().transpose((1, 2, 0))

    mask = M.clone()
    mask[seed] = 190 / 255
    mask = mask.reshape(w_featmap, h_featmap).float()
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0),
                         scale_factor=scales,
                         mode='nearest')[0][0].cpu().numpy()
    return sims, mask


def dino_seg(attn, dims, patch_size, head=0):
    """
    Extraction of boxes based on the DINO segmentation method proposed in https://github.com/facebookresearch/dino. 
    Modified from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
    """
    w_featmap, h_featmap = dims
    nh = attn.shape[1]
    official_th = 0.6

    # We keep only the output patch attention
    # Get the attentions corresponding to [CLS] token
    attentions = attn[0, :, 0, 1:].reshape(nh, -1)

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - official_th)
    idx2 = torch.argsort(idx)
    for h in range(nh):
        th_attn[h] = th_attn[h][idx2[h]]
    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()

    # Connected components
    labeled_array, num_features = scipy.ndimage.label(
        th_attn[head].cpu().numpy())

    # Find the biggest component
    size_components = [
        np.sum(labeled_array == c) for c in range(np.max(labeled_array))
    ]

    if len(size_components) > 1:
        # Select the biggest component avoiding component 0 corresponding to background
        biggest_component = np.argmax(size_components[1:]) + 1
    else:
        # Cases of a single component
        biggest_component = 0

    # Mask corresponding to connected component
    mask = np.where(labeled_array == biggest_component)

    # Add +1 because excluded max
    ymin, ymax = min(mask[0]), max(mask[0]) + 1
    xmin, xmax = min(mask[1]), max(mask[1]) + 1

    # Rescale to image
    r_xmin, r_xmax = xmin * patch_size, xmax * patch_size
    r_ymin, r_ymax = ymin * patch_size, ymax * patch_size
    pred = [r_xmin, r_ymin, r_xmax, r_ymax]

    return pred
