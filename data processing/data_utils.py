

from glob import glob
import pickle
import os
import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
import matplotlib.pyplot as plt
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


def run(pathname):
    filenames_lst = sorted(glob(pathname))
    filenames_lst = [filename.replace("\\", "/") for filename in filenames_lst]

    #base_name = filenames_lst[0].split("/")[-2]
    parent_dir = "/".join(filenames_lst[0].split("/")[:-2])

    #output_pickle_name = os.path.join(parent_dir, base_name + ".pickle")


    images = []
    images_normal_2channel = []
    images_distmap_2channel = []
    for filename in filenames_lst:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #print("image.shape: ", image.shape)
        W = 256
        H = 512
        image = cv2.resize(image, (W, H))
        image = image.reshape((image.shape[0], image.shape[1], -1))
        image = image / 255.0
        image_inv = 1 - image
        images.append(image)
        image = torch.from_numpy(image)
        image_inv = torch.from_numpy(image_inv)
        channel2 = torch.cat((image, image_inv), dim=2)
        channel2 = channel2.numpy()
        #print("channel2.shape: ", channel2.shape)
        distmap = calc_dist_map(channel2)
        images_normal_2channel.append(channel2)
        images_distmap_2channel.append(distmap)

    images = np.asarray(images)
    images_normal_2channel = np.asarray(images_normal_2channel)
    images_distmap_2channel = np.asarray(images_distmap_2channel)
    print("images.shape: ", images.shape)
    print("images_normal_2channel.shape: ", images_normal_2channel.shape)
    print("images_distmap_2channel.shape: ", images_distmap_2channel.shape)
    pickle.dump(images, open(os.path.join(parent_dir, "normalLosses626.pickle"), "wb"))
    pickle.dump(images_normal_2channel, open(os.path.join(parent_dir, "boundaryLosses626.pickle"), "wb"))
    pickle.dump(images_distmap_2channel, open(os.path.join(parent_dir, "boundaryLossesDistmap626.pickle"), "wb"))

def run2(pathname):
    filenames_lst = sorted(glob(pathname))
    filenames_lst = [filename.replace("\\", "/") for filename in filenames_lst]

    parent_dir = "/".join(filenames_lst[0].split("/")[:-2])



    images = []
    for filename in filenames_lst:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        W = 256
        H = 512
        image = cv2.resize(image, (W, H))
        image = image.reshape((image.shape[0], image.shape[1], -1))
        image = image / 255.0
        images.append(image)


    images = np.asarray(images)
    print("images.shape: ", images.shape)
    pickle.dump(images, open(os.path.join(parent_dir, "augmentedTrainData.pickle"), "wb"))

if __name__ == '__main__':
    aug_data = '/env/csc413_project/aug626/data/training/*'
    aug = '/env/csc413_project/aug626/labels/training/*'
    #pickle.dump(pickle_dis_map_2, augmentedTrainDistMap_boundaryLosses)
    #run(aug)
    #run2(aug_data)


    data_train = '/env/csc413_project/aug626/data/normalTraining626.pickle'
    #data_train = '/env/csc413_project/aug626/labels/boundaryLosses626.pickle'
    #data_train = '/env/csc413_project/aug626/labels/normalLosses626.pickle'
    #data_train = '/env/csc413_project/aug626/labels/normalLosses626.pickle'
    imgs = pickle.load(open(data_train, 'rb'))
    print("imgs.shape: ", imgs.shape)
    print(imgs.max(), imgs.min())
    img = imgs[0][:, :, 0]
    # img = torch.from_numpy(img)
    # img = torch.sigmoid(img)
    # img = np.round(img)
    print(img.max(), img.min())
    plt.imshow(img, cmap='gray')
    plt.show()
