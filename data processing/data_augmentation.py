import os
import cv2
import numpy as np
import scipy.io
import pickle
import csv
from skimage.draw import polygon
from scipy.ndimage import distance_transform_edt
import torch
from torch import Tensor
from typing import cast
from scipy.ndimage import distance_transform_edt as eucl_distance
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as distance

# desired image width and height
img_width = 256
img_height = 512

# Supporting function(s)

def adjust_gamma(image, gamma):
    """
    By Adrian Rosebrock
    on October 5, 2015
    https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def convert_label_1(label):
    """
    Converting ground truth label from list of pixel coordinates of landmarks to a 2D array with size equals to the size
    of input image, where value set to 1 when the corresponding pixel is a landmark otherwise 0.
    """
    label[:, 0] = np.clip(label[:, 0], 0, img_width - 1)
    label[:, 1] = np.clip(label[:, 1], 0, img_height - 1)

    keymap = np.zeros((img_height, img_width))
    for i in range(17):
        cv2.fillPoly(keymap, pts=np.array([[[label[4 * i][0], label[4 * i][1]], [label[4 * i + 1][0], label[4 * i + 1][1]],
                                           [label[4 * i + 3][0], label[4 * i + 3][1]], [label[4 * i + 2][0], label[4 * i + 2][1]]]], dtype=np.int32), color=1)
    return keymap

def convert_label_2(label):
    label[:, 0] = np.clip(label[:, 0], 0, img_width - 1)
    label[:, 1] = np.clip(label[:, 1], 0, img_height - 1)

    keymap = np.zeros((img_height, img_width, 2))
    keymap[:, :, 1] = keymap[:, :, 1] + 1
    for i in range(17):
        rr, cc = polygon([label[4 * i][1], label[4 * i + 1][1], label[4 * i + 3][1], label[4 * i + 2][1]],
                         [label[4 * i][0], label[4 * i + 1][0], label[4 * i + 3][0], label[4 * i + 2][0]])

        keymap[rr, cc, 0] = 1
        keymap[rr, cc, 1] = 0
    return keymap


def calc_dist_map(label):
    res = np.zeros_like(label)
    posmask = label[:, :, 0].astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res[:, :, 0] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    posmask = label[:, :, 1].astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res[:, :, 1] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

##########################################################################################

# Input data directories
train_data_dir = "/env/csc413_project/boostnet_labeldata/data/training/"
train_label_dir = "/env/csc413_project/boostnet_labeldata/labels/training/"
train_label_csv_name = "/env/csc413_project/boostnet_labeldata/labels/training/landmarks.csv"
train_filename_dir = "/env/csc413_project/boostnet_labeldata/labels/training/filenames.csv"
# Output directories
train_augmented_data_dir = "/env/csc413_project/boostnet_aug_new/data/augmentedTraining"
train_augmented_label_dir = "/env/csc413_project/boostnet_aug_new/labels/augmentedTraining"
pickle_data_out = "/env/csc413_project/boostnet_aug_new/data"
pickle_label_out = "/env/csc413_project/boostnet_aug_new/labels"

# Data Augmentation
#   - mirroring
#   - tilting (small angles only)
#   - adjusting gamma
#   Expand the data set, not considering large angle tilting because it won't happens in real life.

pickle_data = []
pickle_label = []
pickle_label_2 = []
pickle_dis_map_2 = []


filenames_lst = open(train_filename_dir, 'r').read().split("\n")[:-1]

landmark = list(csv.reader(open(train_label_csv_name)))
for i in range(len(landmark)):
  for j in range(136):
    landmark[i][j] = float(landmark[i][j])

index = -1
sum_vert = 0
sum_conn = 0
for filename in filenames_lst:
    index += 1
    curr_label = landmark[index]
    corresponding_label = np.asarray([curr_label[:68], curr_label[68:]]).T
    corresponding_label[:, 0] = corresponding_label[:, 0] * img_width
    corresponding_label[:, 1] = corresponding_label[:, 1] * img_height

    corresponding_data_dir = os.path.join(train_data_dir, filename)
    corresponding_label_dir = os.path.join(train_label_dir, filename)

    extracted_image = cv2.imread(corresponding_data_dir, cv2.IMREAD_GRAYSCALE)
    corresponding_image = cv2.resize(extracted_image, (img_width, img_height))

    # store original image and label to output directory
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename), corresponding_image)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]), corresponding_label)

    # --------------------------------adjusting gamma (landmark locations not changed)-------------------------------- #
    gamma_image_b = adjust_gamma(corresponding_image, 1.3)

    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " gamma adjusted B.jpg", gamma_image_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " gamma adjusted B", corresponding_label)

    # ------------------------------------mirroring (landmark locations mirrored)------------------------------------ #
    mirrored_image = cv2.flip(corresponding_image, 1)

    im_width = corresponding_image.shape[1]
    mirrored_label = np.asarray(np.concatenate(
        (np.matrix(im_width - corresponding_label[:, 0]).T, np.matrix(corresponding_label[:, 1]).T), axis=1))

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " mirrored.jpg", mirrored_image)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " mirrored", mirrored_label)

    # adjusting gamma for mirrored samples
    gamma_mirrored_image_b = adjust_gamma(mirrored_image, 1.3)

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " mirrored gamma adjusted B.jpg",
                gamma_mirrored_image_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " mirrored gamma adjusted B", mirrored_label)

    # -------------------------------------- tilting (landmark locations tilted)-------------------------------------- #
    img_center = (corresponding_image.shape[1] / 2, corresponding_image.shape[0] / 2)
    label_add_one_transposed = np.append(corresponding_label.T, [np.ones(corresponding_label.shape[0])], axis=0)

    rotation_matrix_a = cv2.getRotationMatrix2D(img_center, np.random.randint(-5, 0), 1)
    rotation_matrix_b = cv2.getRotationMatrix2D(img_center, np.random.randint(1, 6), 1)

    tilted_image_a = cv2.warpAffine(corresponding_image, rotation_matrix_a,
                                    (corresponding_image.shape[1], corresponding_image.shape[0]))

    tilted_label_a = np.floor(np.dot(rotation_matrix_a, label_add_one_transposed).T).astype(int)

    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted A.jpg", tilted_image_a)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted A", tilted_label_a)

    # adjusting gamma for tilted samples
    gamma_tilted_image_a_b = adjust_gamma(tilted_image_a, 1.3)

    # store new images and labels
    cv2.imwrite(os.path.join(train_augmented_data_dir, filename[:-4]) + " tilted A gamma adjusted B.jpg",
                gamma_tilted_image_a_b)
    np.save(os.path.join(train_augmented_label_dir, filename[:-4]) + " tilted A gamma adjusted B", tilted_label_a)

    # ---------------------------------------------tilting and mirroring--------------------------------------------- #
    mirrored_label_add_one_transposed = np.append(mirrored_label.T, [np.ones(corresponding_label.shape[0])], axis=0)

    rotation_matrix_c = cv2.getRotationMatrix2D(img_center, np.random.randint(-5, 0), 1)
    rotation_matrix_d = cv2.getRotationMatrix2D(img_center, np.random.randint(1, 6), 1)

    pickle_data.extend([corresponding_image / 255,
                        gamma_image_b / 255,
                        mirrored_image / 255,
                        gamma_mirrored_image_b / 255,
                        tilted_image_a / 255,
                        gamma_tilted_image_a_b / 255,
                       ])


    converted_corresponding_label_1 = convert_label_1(corresponding_label)
    converted_mirrored_label_1 = convert_label_1(mirrored_label)
    converted_tilted_label_a_1 = convert_label_1(tilted_label_a)

    converted_corresponding_label_2 = convert_label_2(corresponding_label)
    converted_mirrored_label_2 = convert_label_2(mirrored_label)
    converted_tilted_label_a_2 = convert_label_2(tilted_label_a)


    pickle_label.extend(
        [converted_corresponding_label_1,
         converted_corresponding_label_1,
         converted_mirrored_label_1,
         converted_mirrored_label_1,
         converted_tilted_label_a_1,
         converted_tilted_label_a_1,
         ])

    pickle_label_2.extend(
        [converted_corresponding_label_2,
         converted_corresponding_label_2,
         converted_mirrored_label_2,
         converted_mirrored_label_2,
         converted_tilted_label_a_2,
         converted_tilted_label_a_2,
         ])

# dist map for the boundary loss
    for item in [converted_corresponding_label_2, converted_mirrored_label_2, converted_tilted_label_a_2]:
        dist_map_target = calc_dist_map(item)

        pickle_dis_map_2.append(dist_map_target)
        pickle_dis_map_2.append(dist_map_target)

pickle_data = np.array(pickle_data).reshape(-1, img_height, img_width, 1).astype(np.float32)
pickle_label = np.array(pickle_label).astype(np.float32)
pickle_label_2 = np.array(pickle_label_2).astype(np.float32)
pickle_dis_map_2 = np.array(pickle_dis_map_2).astype(np.float32)

print("Preprocess finished! now storing data/label...")

# store pickle data and label
augmentedTrainData = open(os.path.join(pickle_data_out, "augmentedTrainData.pickle"), "wb")
pickle.dump(pickle_data, augmentedTrainData)
augmentedTrainData.close()

augmentedTrainLabel_normalLosses = open(os.path.join(pickle_label_out, "augmentedTrainLabel_normalLosses.pickle"), "wb")
pickle.dump(pickle_label, augmentedTrainLabel_normalLosses)
augmentedTrainLabel_normalLosses.close()

augmentedTrainLabel_boundaryLosses = open(os.path.join(pickle_label_out, "augmentedTrainLabel_boundaryLosses.pickle"), "wb")
pickle.dump(pickle_label_2, augmentedTrainLabel_boundaryLosses)
augmentedTrainLabel_boundaryLosses.close()

augmentedTrainDistMap_boundaryLosses = open(os.path.join(pickle_label_out, "augmentedTrainDistMap_boundaryLosses.pickle"), "wb")
pickle.dump(pickle_dis_map_2, augmentedTrainDistMap_boundaryLosses)
augmentedTrainDistMap_boundaryLosses.close()
