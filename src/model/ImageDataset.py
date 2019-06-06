import torch
import numpy as np
from scipy.io import loadmat


def load_all_patches(N:int, F:int, border:int=4) -> torch.tensor:
    '''
    Create (num_images x N) number of patches of size F by F.

    Args:
        N (int): number of patches to crop per image
        F (int): size of (sqaure) patches
        border (int, optional): the border to exclude from cropping in images

    Returns:
        cropped image patches of shape ((num_images x N), F ** 2)
    '''
    # load mat
    X = loadmat("../../data/IMAGES.mat")
    X = X['IMAGES']
    img_size = X.shape[0]
    n_img = X.shape[2]
    images = torch.zeros((N * n_img, F ** 2))
    # for every image
    counter = 0
    for i in range(n_img):
        img = X[:,:,i]
        for j in range(N):
            x = np.random.randint(border, img_size-F-border)
            y = np.random.randint(border, img_size-F-border)
            crop = torch.tensor(img[x:x+F, y:y+F]).view(-1,)
            images[counter, :] = crop - crop.mean()
            counter += 1
    return images

