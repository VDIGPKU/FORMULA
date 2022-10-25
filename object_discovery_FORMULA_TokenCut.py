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

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage
import cv2


def FORMULA(feats,
         dims,
         scales,
         init_image_size,
         tau=0,
         eps=1e-5,
         im_name='',
         no_binary_graph=False):
    """
    Implementation of FORMULA Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0, 0:1, :].cpu().numpy()

    feats = feats[0, 1:, :]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1, 0))

    A = A.cpu().numpy()
    if no_binary_graph:
        A[A < tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)

    # Print second and third smallest eigenvector
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])
    eigenvec = np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition
    second_smallest_vec = eigenvectors[:, 0]

    for _ in range(4):
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        seed = np.argmax(np.abs(second_smallest_vec))

        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
        bipartition = bipartition.reshape(dims).astype(float)

        Y, X = np.where(bipartition == 1)  # row, col

        center_r, center_c = np.sum(Y) / len(Y), np.sum(X) / len(X)

        t = int(np.sqrt(len(X)) * 1)
        t = min(dims) if min(dims) < t else t
        sigma = 1
        x_gauss = cv2.getGaussianKernel(t, sigma)
        xy_kernel = np.multiply(x_gauss.T, x_gauss)

        fb = np.zeros(dims)

        r_start, r_end = int(center_r - t / 2), int(center_r - t / 2) + t
        c_start, c_end = int(center_c - t / 2), int(center_c - t / 2) + t


        if r_start < 0:
            r_start = 0
            r_end = t
        if c_start < 0:
            c_start = 0
            c_end = t
        if r_end > dims[0]:
            r_end = dims[0]
            r_start = r_end - t
        if c_end > dims[1]:
            c_end = dims[1]
            c_start = c_end - t

        fb[r_start:r_end, c_start:c_end] = xy_kernel
        fb = (fb + 1).reshape(dims[0] * dims[1])

        second_smallest_vec *= fb

    # predict BBox
    pred, _, objects, cc = detect_box(
        bipartition,
        seed,
        dims,
        scales=scales,
        initial_im_size=init_image_size[1:]
    )  ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0], cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)


def detect_box(bipartition,
               seed,
               dims,
               initial_im_size=None,
               scales=None,
               principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]

    if principle_object:
        mask = np.where(objects == cc)
        # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError
