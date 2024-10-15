
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import read_pickle_file
import matplotlib.pyplot as plt
import glob
class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, dist_map_path=None, image_size=(256, 512), image_channel=1):

        self.images_path = images_path
        self.masks_path = masks_path
        self.dist_map_path = dist_map_path
        self.n_samples = len(images_path)
        self.image_channel = image_channel
        self.image_size = image_size

    def __getitem__(self, index):
        """ Reading image """
        if (self.image_channel == 3):
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        size = self.image_size

        image = cv2.resize(image, size)
        image = image.reshape((image.shape[0], image.shape[1], -1))
        image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        mask = mask/255.0   ## (512, 512)
        mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        #plt.imshow(mask[0], cmap='gray')
        #plt.show()
        if self.dist_map_path is None:
            return image, mask
        else:
            """Reading Dist_Map for the Boundary Loss"""
            map = cv2.imread(self.dist_map_path[index], cv2.IMREAD_GRAYSCALE)
            map = cv2.resize(map, size)
            # map = map / 255.0  ## (512, 512)
            map = np.expand_dims(map, axis=0)  ## (1, 512, 512)
            map = map.astype(np.float32)
            map = torch.from_numpy(map)

            return image, mask, map

    def __len__(self):
        return self.n_samples


class DriveDatasetPickle(Dataset):
    def __init__(self, image_pickle_path, mask_pickle_path, dist_map_pickle_path=None, image_size=(256, 512)):
        self.image_pickle = read_pickle_file(image_pickle_path)
        self.mask_pickle = read_pickle_file(mask_pickle_path)
        samples, height, width, channels = self.image_pickle.shape
        self.boundary_loss = False
        if dist_map_pickle_path is not None:
            self.boundary_loss = True
            #print("Loading Distance Map from pickle file", dist_map_pickle_path)
            self.dist_map_pickle = read_pickle_file(dist_map_pickle_path)
            #print("Distance Map Loaded")
        else:
            self.dist_map_pickle = None
        self.n_samples = samples
        self.image_size = image_size
    def fix_mask(self, mask):
        mask = torch.from_numpy(mask)
        mask = torch.sigmoid(mask)
        mask = np.round(mask)
        mask = mask.numpy()
        return mask
    def __getitem__(self, index):
        """ Reading image """
        #print("Pickle Image getitem")
        image = self.image_pickle[index]
        size = self.image_size

        image = cv2.resize(image, size)
        image = image.reshape((image.shape[0], image.shape[1], -1))
        #print(image.max(), image.min())
        #image = image/255.0 ## (512, 512, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        #plt.imshow(image[0], cmap='gray')
        #plt.show()
        """ Reading mask """
        mask = self.mask_pickle[index]
        #mask = self.fix_mask(mask)
        mask = cv2.resize(mask, size)


        #print(mask.max(), mask.min())
        #mask = mask/255.0   ## (512, 512)
        mask = mask.reshape(-1, mask.shape[0], mask.shape[1])
        #mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        #print("image shape", image.shape)
        #print("mask shape" ,mask.shape)
        mask = mask.astype(np.float32)
        #plt.imshow(mask[0], cmap='gray')
        #plt.show()
        if self.dist_map_pickle is None:
            return image, mask
        else:
            """Reading Dist_Map for the Boundary Loss"""
            #print("Pickle Dist Map")
            map = self.dist_map_pickle[index]
            #print("map shape", map.shape)
            map = cv2.resize(map, size)
            # map = map / 255.0  ## (512, 512)
            map = np.expand_dims(map, axis=0)  ## (1, 512, 512)
            map = map.astype(np.float32)
            map = torch.from_numpy(map)

            return image, mask, map

    def __len__(self):
        return self.n_samples


class BoundaryLossDatasetPickle(Dataset):
    def __init__(self, image_pickle_path, mask_pickle_path, dist_map_pickle_path=None, image_size=(256, 512)):
        self.image_pickle = read_pickle_file(image_pickle_path)
        self.mask_pickle = read_pickle_file(mask_pickle_path)
        samples, height, width, channels = self.image_pickle.shape
        if dist_map_pickle_path is not None:
            #print("Loading Distance Map from pickle file", dist_map_pickle_path)
            self.dist_map_pickle = read_pickle_file(dist_map_pickle_path)
            #print("Distance Map Loaded")
        else:
            self.dist_map_pickle = None
        self.n_samples = samples
        self.image_size = image_size
    def fix_mask(self, mask):
        mask = torch.from_numpy(mask)
        mask = torch.sigmoid(mask)
        mask = np.round(mask)
        mask = mask.numpy()
        return mask
    def __getitem__(self, index):
        """ Reading image """
        #print("Pickle Image getitem")
        image = self.image_pickle[index]
        size = self.image_size

        image = cv2.resize(image, size)
        image = image.reshape((image.shape[0], image.shape[1], -1))
        image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        #image = torch.cat((image, image), dim=0)
        """ Reading mask """
        mask = self.mask_pickle[index]
        #mask = cv2.resize(mask, size)

        mask = np.transpose(mask, (2, 0, 1))
        #mask = np.expand_dims(mask, axis=0) ## (1, 512, 512)
        mask = mask.astype(np.float32)
        #plt.imshow(mask[0], cmap='gray')
        #plt.show()
        if self.dist_map_pickle is None:
            return image, mask
        else:
            """Reading Dist_Map for the Boundary Loss"""
            #print("Pickle Dist Map")
            map = self.dist_map_pickle[index]
            map = np.transpose(map, (2, 0, 1))
            #print(map.shape)
            #map = cv2.resize(map, size)
            # map = map / 255.0  ## (512, 512)
            #map = np.expand_dims(map, axis=0)  ## (1, 512, 512)
            map = map.astype(np.float32)
            map = torch.from_numpy(map)
            #print("image shape", image.shape)
            #print("mask shape", mask.shape)
            #print("map shape", map.shape)
            return image, mask, map

    def __len__(self):
        return self.n_samples

if __name__ == '__main__':

    #label = read_pickle_file('/env/csc413_project/boostnet_fixed/labels/augmentedTrainDC128by352_3.pickle')

    filename = '/env/csc413_project/boostnet_aug_new/labels/augmentedTrainDistMap_boundaryLosses.pickle'
    filename = '/env/csc413_project/boostnet_aug_new/labels/augmentedTrainLabel_normalLosses.pickle'
    #filename = '/env/csc413_project/boostnet_aug_new/data/augmentedTrainData.pickle'
    label = read_pickle_file(filename)

    #label = read_pickle_file('/env/csc413_project/boostnet_labeldata_luke_aug/data/augmentedTrainData128by352.pickle')
    #for i in range(2886):
        #img = label[i]
        #print(i, img)

    files = sorted( glob.glob('/env/csc413_project/boostnet_aug_new/data/augmentedTraining/*'))
    for i, f in enumerate(files):
        # get base name
        basename = os.path.basename(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img[label[i] == 1] = 255
        cv2.imwrite('/env/csc413_project/boostnet_aug_new/data/augmentedTrainingMaskMerged/' + basename, img)


    print(os.path.basename(filename), label.shape)
    img = label[0]#[:,:,0]
    #img = torch.from_numpy(img)
    #img = torch.sigmoid(img)
    #img = np.round(img)
    print(img.max(), img.min())
    plt.imshow(img, cmap='gray')
    plt.show()
