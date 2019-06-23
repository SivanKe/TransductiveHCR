import numpy as np
import cv2
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.color import hsv2rgb
import scipy
from os.path import join, basename
import os
import glob
import random
from random import shuffle

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        return sample


class ElasticAndSine(object):

    def __init__(self, elastic_alpha, elastic_sigma, sin_magnitude, random_state=None):
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.sin_magnitude = sin_magnitude
        self.random_state = random_state

    def __elastic_and_sin_transform(self, image):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        image = image.astype(float) / 255.

        if self.random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

        # add sinusoidal oscillation to row coordinates
        start = np.random.rand(1) * np.pi * 0.5
        end = start + (np.random.rand(1) * np.pi) + 1e-6
        magnitude = np.random.rand(1) * self.sin_magnitude
        sin_line = - np.sin(np.linspace(start, end, shape[1])) * magnitude
        sinus_change = (sin_line - np.ceil(np.max(sin_line))).reshape(1,-1,1)

        indices = np.reshape(y+dy+sinus_change, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        distored_image = (distored_image * 255).astype('uint8')
        return distored_image

    def __call__(self, sample):
        sample["img"] = self.__elastic_and_sin_transform(sample["img"])
        return sample

class OnlySine(object):

    def __init__(self, sin_magnitude, random_state=None):
        self.sin_magnitude = sin_magnitude
        self.random_state = random_state

        map_coordinates

    def __call__(self, sample):
        sample["img"] = self.__sin_transform(sample["img"])
        return sample


class OnlyElastic(object):

    def __init__(self, elastic_alpha, elastic_sigma, random_state=None):
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.random_state = random_state

    def __elastic_and_sin_transform(self, image):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        image = image.astype(float) / 255.

        if self.random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.elastic_sigma, mode="constant", cval=0) * self.elastic_alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

        # add sinusoidal oscillation to row coordinates

        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
        distored_image = (distored_image * 255).astype('uint8')
        return distored_image

    def __call__(self, sample):
        sample["img"] = self.__elastic_and_sin_transform(sample["img"])
        return sample
'''
class RandomErasion(object):
    def __init__(self, image_precentage, erase_precentage):
        self.image_precentage = image_precentage
        self.erase_precentage = erase_precentage
    def __call__(self, sample):
        # need to deal with rgb!!!
        sample_one_dim = sample[:, :, 0].reshape(sample.shape[-1])
        non_zero_pixels = np.argwhere(sample_one_dim > 0)
        erase_indices = range(len(non_zero_pixels))
        shuffle(erase_indices)
        erase_indices[:int(self.erase_precentage * len(erase_indices))]
'''

class Resize(object):
    def __init__(self, hight=64):
        self.hight = hight

    def __call__(self, sample):
        aspect_ratio = self.hight/sample["img"].shape[0]
        if aspect_ratio != 1:
            old_shape = sample["img"].shape
            sample["img"] = scipy.misc.imresize(sample["img"], (self.hight, int(aspect_ratio * sample["img"].shape[1])), interp='lanczos')
            #sample["img"] = cv2.resize(sample["img"], dsize=(self.hight,int(aspect_ratio * sample["img"].shape[1])))

        return sample

class AddWidth(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["img_width"] = sample["img"].shape[1]
        return sample

def rotate_bound(image, angle, fill_value):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=fill_value)


class Rotation(object):
    def __init__(self, angle=20, fill_value=0, p = 0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) > self.p:
            return sample
        h,w,_ = sample["img"].shape
        ang_rot = np.random.uniform(self.angle) - self.angle / 2
        sample["img"] = rotate_bound(sample["img"], ang_rot,  [self.fill_value,self.fill_value,self.fill_value])

        #transform = cv2.getRotationMatrix2D((w/2, h/2), ang_rot, 1)
        #sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample

class Translation(object):
    def __init__(self, fill_value=0, p = 0.5):
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h,w,_ = sample["img"].shape
        trans_range = [w / 10, h / 10]
        tr_x = trans_range[0]*np.random.uniform()-trans_range[0]/2
        tr_y = trans_range[1]*np.random.uniform()-trans_range[1]/2
        transform = np.float32([[1,0, tr_x], [0,1, tr_y]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample

class Scale(object):
    def __init__(self, scale=[0.5, 1.2], fill_value=0, p = 0.5):
        self.scale = scale
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        scale = np.random.uniform(self.scale[0], self.scale[1])
        transform = np.float32([[scale, 0, 0],[0, scale, 0]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample

"""
class SimplexNoise(object):

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) > self.p:
            return sample
        sample["img"] = self.__simplex_bg(sample["img"])
        return sample

    @staticmethod
    def __gradient_background(image, sin_start_range, sin_size_range, min_value, axis=1):
        func_size = image.shape[axis]
        sin_start = (np.random.rand(1) * (sin_start_range[1] - sin_start_range[0])) + sin_start_range[0]
        sin_size = (np.random.rand(1) * (sin_size_range[1] - sin_size_range[0])) + sin_size_range[0]

        value_change_range = [0.01, 0.2]
        value_change = min(
            (np.random.rand(1) * (value_change_range[1] - value_change_range[0])) + value_change_range[0],
            (1. - min_value))
        sin_range = np.linspace(sin_start, sin_start + sin_size, func_size)
        background = (np.sin(sin_range) * value_change) + min_value

        shp = [-1, -1]
        shp[1 - axis] = 1
        rep = [1, 1]
        rep[1 - axis] = image.shape[1 - axis]
        background = np.tile(background.reshape(shp), rep)
        return background

    @staticmethod
    def __color_aug(image):
        '''
        image : ndarray
        Input image data. Will be converted to float.
        mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        :param noise_type:
        :param image:
        :return:
        '''
        bg_grad_axis = np.random.randint(2)
        sin_start_range = [0, np.random.rand(1) * np.pi]
        sin_size_range = [0.02, 2 * np.pi]

        bg_base = ColorGradGausNoise.__gradient_background(image, sin_start_range, sin_size_range, 0, axis=bg_grad_axis)
        row, col = bg_base.shape
        h_base_range = [0. / 360., 360.0 / 360.]
        base_h = (np.random.rand() * (h_base_range[1] - h_base_range[0])) + h_base_range[0]
        h_var_renge = [0.01 / 360., 1. / 360.]
        var_h = (np.random.rand() * (h_var_renge[1] - h_var_renge[0])) + h_var_renge[0]
        s_base_range = [01. / 100., 30. / 100.]
        base_s = (np.random.rand() * (s_base_range[1] - s_base_range[0])) + s_base_range[0]
        s_var_renge = [0.05 / 100., 1. / 100.]
        var_s = (np.random.rand() * (s_var_renge[1] - s_var_renge[0])) + s_var_renge[0]
        v_var_renge = [0.01 / 100., 1. / 100.]
        v_min_value_range = [0.7, 0.95]
        v_min_value = (np.random.rand(1) * (v_min_value_range[1] - v_min_value_range[0])) + v_min_value_range[0]
        var_v = (np.random.rand() * (v_var_renge[1] - v_var_renge[0])) + v_var_renge[0]
        var = np.array([var_h, var_s, var_v])
        sigma = var ** 0.5
        gauss_h = np.clip(np.random.normal(base_h, sigma[0], (row, col)).reshape(row, col), 0, 1)
        gauss_s = np.clip(np.random.normal(base_s, sigma[1], (row, col)).reshape(row, col) - bg_base, 0, 1)
        gauss_v = np.clip(np.random.normal(v_min_value, sigma[2], (row, col)).reshape(row, col) + bg_base, 0, 1)
        gauss_hsv = np.stack([gauss_h, gauss_s, gauss_v], axis=2)
        gauss = (hsv2rgb(gauss_hsv) * 255).astype(int)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        # image[image == 255] = gauss[image == 255]
        return np.clip(gauss - (255 - image), 0, 255)

class ColorGausOrSimplexNoise(object):

    def __init__(self, p=0.7):
        self.p = 1#p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) > self.p:
            return sample
        if np.random.uniform(0.0, 1.0) > 0.0:
            sample["img"] = self.__simplex_bg(sample["img"])
        else:
            sample["img"] = self.__simplex_bg(sample["img"])
        return sample

    @staticmethod
    def __simplex_bg(img):
        img = img.astype(np.int32)
        h, w, _ = img.shape
        octaves = [1]
        min_border = 30 / len(octaves)
        max_border = 150 / len(octaves)
        noise_im = np.zeros_like(img)
        for octave in octaves:
            freq = 16.0 * octave
            min_val = random.randint(0, min_border)
            max_val = random.randint(max(min_val + 10, 0), max_border)
            for y in range(h):
                for x in range(w):
                    noise_im[y, x, :] += int(((snoise2(x / freq, y / freq, octave) + 1) / 2) * (max_val - min_val)) + min_val

        img = np.clip(img - noise_im, 0, 255)
        img = img.astype(np.uint8)
        '''
        simplex = OpenSimplex()
        fs_options = range(4, 32, 4)
        FEATURE_SIZE = float(fs_options[np.random.randint(0, len(fs_options))])
        noise_im = np.zeros_like(img)
        h, w, _ = img.shape
        min_val = random.randint(-30, 30)
        max_val = random.randint(max(min_val + 10, 0), 220)
        for y in range(0, h):
            for x in range(0, w):
                noise_im[y, x, :] = int(
                    ((simplex.noise2d(x / FEATURE_SIZE, y / FEATURE_SIZE) + 1) / 2) * (max_val - min_val)) + min_val
        img = np.clip(noise_im + img, 0, 255)
        '''
        return img

    @staticmethod
    def __color_aug(image):
        '''
        image : ndarray
        Input image data. Will be converted to float.
        mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        :param noise_type:
        :param image:
        :return:
        '''
        bg_grad_axis = np.random.randint(2)
        sin_start_range = [0, np.random.rand(1) * np.pi]
        sin_size_range = [0.02, 2 * np.pi]

        row, col = image.shape[:2]
        h_base_range = [0. / 360., 360.0 / 360.]
        base_h = (np.random.rand() * (h_base_range[1] - h_base_range[0])) + h_base_range[0]
        h_var_renge = [0.01 / 360., 1. / 360.]
        var_h = (np.random.rand() * (h_var_renge[1] - h_var_renge[0])) + h_var_renge[0]
        s_base_range = [01. / 100., 30. / 100.]
        base_s = (np.random.rand() * (s_base_range[1] - s_base_range[0])) + s_base_range[0]
        s_var_renge = [0.05 / 100., 1. / 100.]
        var_s = (np.random.rand() * (s_var_renge[1] - s_var_renge[0])) + s_var_renge[0]
        v_var_renge = [0.01 / 100., 1. / 100.]
        v_min_value_range = [0.7, 0.95]
        v_min_value = (np.random.rand(1) * (v_min_value_range[1] - v_min_value_range[0])) + v_min_value_range[0]
        var_v = (np.random.rand() * (v_var_renge[1] - v_var_renge[0])) + v_var_renge[0]
        var = np.array([var_h, var_s, var_v])
        sigma = var ** 0.5
        gauss_h = np.clip(np.random.normal(base_h, sigma[0], (row, col)).reshape(row, col), 0, 1)
        gauss_s = np.clip(np.random.normal(base_s, sigma[1], (row, col)).reshape(row, col), 0, 1)
        gauss_v = np.clip(np.random.normal(v_min_value, sigma[2], (row, col)).reshape(row, col), 0, 1)
        gauss_hsv = np.stack([gauss_h, gauss_s, gauss_v], axis=2)
        gauss = (hsv2rgb(gauss_hsv) * 255).astype(int)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        # image[image == 255] = gauss[image == 255]
        return np.clip(gauss - (255 - image), 0, 255)
"""

class ColorGausNoise(object):

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) > self.p:
            return sample
        sample = self.__color_aug(sample)
        return sample

    @staticmethod
    def __color_aug(image):
        '''
        image : ndarray
        Input image data. Will be converted to float.
        mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        :param noise_type:
        :param image:
        :return:
        '''
        row, col = image.shape[:2]
        h_base_range = [0. / 360., 360.0 / 360.]
        base_h = (np.random.rand() * (h_base_range[1] - h_base_range[0])) + h_base_range[0]
        h_var_renge = [0.01 / 360., 1. / 360.]
        var_h = (np.random.rand() * (h_var_renge[1] - h_var_renge[0])) + h_var_renge[0]
        s_base_range = [01. / 100., 30. / 100.]
        base_s = (np.random.rand() * (s_base_range[1] - s_base_range[0])) + s_base_range[0]
        s_var_renge = [0.05 / 100., 1. / 100.]
        var_s = (np.random.rand() * (s_var_renge[1] - s_var_renge[0])) + s_var_renge[0]
        v_var_renge = [0.01 / 100., 1. / 100.]
        v_min_value_range = [0.7, 0.95]
        v_min_value = (np.random.rand(1) * (v_min_value_range[1] - v_min_value_range[0])) + v_min_value_range[0]
        var_v = (np.random.rand() * (v_var_renge[1] - v_var_renge[0])) + v_var_renge[0]
        var = np.array([var_h, var_s, var_v])
        sigma = var ** 0.5
        gauss_h = np.clip(np.random.normal(base_h, sigma[0], (row, col)).reshape(row, col), 0, 1)
        gauss_s = np.clip(np.random.normal(base_s, sigma[1], (row, col)).reshape(row, col), 0, 1)
        gauss_v = np.clip(np.random.normal(v_min_value, sigma[2], (row, col)).reshape(row, col), 0, 1)
        gauss_hsv = np.stack([gauss_h, gauss_s, gauss_v], axis=2)
        gauss = (hsv2rgb(gauss_hsv) * 255).astype(int)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        # image[image == 255] = gauss[image == 255]
        return np.clip(gauss - (255 - image), 0, 255)


class ColorGradGausNoise(object):

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) > self.p:
            return sample
        sample["img"] = self.__color_aug(sample["img"])
        return sample

    @staticmethod
    def __gradient_background(image, sin_start_range, sin_size_range, min_value, axis=1):
        func_size = image.shape[axis]
        sin_start = (np.random.rand(1) * (sin_start_range[1] - sin_start_range[0])) + sin_start_range[0]
        sin_size = (np.random.rand(1) * (sin_size_range[1] - sin_size_range[0])) + sin_size_range[0]

        value_change_range = [0.01, 0.2]
        value_change = min(
            (np.random.rand(1) * (value_change_range[1] - value_change_range[0])) + value_change_range[0],
            (1. - min_value))
        sin_range = np.linspace(sin_start, sin_start + sin_size, func_size)
        background = (np.sin(sin_range) * value_change) + min_value

        shp = [-1, -1]
        shp[1 - axis] = 1
        rep = [1, 1]
        rep[1 - axis] = image.shape[1 - axis]
        background = np.tile(background.reshape(shp), rep)
        return background

    @staticmethod
    def __color_aug(image):
        '''
        image : ndarray
        Input image data. Will be converted to float.
        mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        :param noise_type:
        :param image:
        :return:
        '''
        bg_grad_axis = np.random.randint(2)
        sin_start_range = [0, np.random.rand(1) * np.pi]
        sin_size_range = [0.02, 2 * np.pi]

        bg_base = ColorGradGausNoise.__gradient_background(image, sin_start_range, sin_size_range, 0, axis=bg_grad_axis)
        row, col = bg_base.shape
        h_base_range = [0. / 360., 360.0 / 360.]
        base_h = (np.random.rand() * (h_base_range[1] - h_base_range[0])) + h_base_range[0]
        h_var_renge = [0.01 / 360., 1. / 360.]
        var_h = (np.random.rand() * (h_var_renge[1] - h_var_renge[0])) + h_var_renge[0]
        s_base_range = [01. / 100., 30. / 100.]
        base_s = (np.random.rand() * (s_base_range[1] - s_base_range[0])) + s_base_range[0]
        s_var_renge = [0.05 / 100., 1. / 100.]
        var_s = (np.random.rand() * (s_var_renge[1] - s_var_renge[0])) + s_var_renge[0]
        v_var_renge = [0.01 / 100., 1. / 100.]
        v_min_value_range = [0.7, 0.95]
        v_min_value = (np.random.rand(1) * (v_min_value_range[1] - v_min_value_range[0])) + v_min_value_range[0]
        var_v = (np.random.rand() * (v_var_renge[1] - v_var_renge[0])) + v_var_renge[0]
        var = np.array([var_h, var_s, var_v])
        sigma = var ** 0.5
        gauss_h = np.clip(np.random.normal(base_h, sigma[0], (row, col)).reshape(row, col), 0, 1)
        gauss_s = np.clip(np.random.normal(base_s, sigma[1], (row, col)).reshape(row, col), 0, 1)
        gauss_v = np.clip(np.random.normal(v_min_value, sigma[2], (row, col)).reshape(row, col), 0, 1)
        gauss_hsv = np.stack([gauss_h, gauss_s, gauss_v], axis=2)
        gauss = (hsv2rgb(gauss_hsv) * 255).astype(int)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        # image[image == 255] = gauss[image == 255]
        return np.clip(gauss - (255 - image), 0, 255)


class ColorGrad(object):

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) > self.p:
            return sample
        sample["img"] = self.__color_aug(sample["img"])
        return sample

    @staticmethod
    def __gradient_background(image, sin_start_range, sin_size_range, min_value, axis=1):
        func_size = image.shape[axis]
        sin_start = (np.random.rand(1) * (sin_start_range[1] - sin_start_range[0])) + sin_start_range[0]
        sin_size = (np.random.rand(1) * (sin_size_range[1] - sin_size_range[0])) + sin_size_range[0]

        value_change_range = [0.01, 0.2]
        value_change = min(
            (np.random.rand(1) * (value_change_range[1] - value_change_range[0])) + value_change_range[0],
            (1. - min_value))
        sin_range = np.linspace(sin_start, sin_start + sin_size, func_size)
        background = (np.sin(sin_range) * value_change) + min_value

        shp = [-1, -1]
        shp[1 - axis] = 1
        rep = [1, 1]
        rep[1 - axis] = image.shape[1 - axis]
        background = np.tile(background.reshape(shp), rep)
        return background

    @staticmethod
    def __color_aug(image):
        '''
        image : ndarray
        Input image data. Will be converted to float.
        mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
        :param noise_type:
        :param image:
        :return:
        '''
        bg_grad_axis = np.random.randint(2)
        sin_start_range = [0, np.random.rand(1) * np.pi]
        sin_size_range = [0.02, 2 * np.pi]

        bg_base = ColorGrad.__gradient_background(image, sin_start_range, sin_size_range, 0, axis=bg_grad_axis)
        row, col = bg_base.shape
        h_base_range = [0. / 360., 360.0 / 360.]
        base_h = (np.random.rand() * (h_base_range[1] - h_base_range[0])) + h_base_range[0]
        h_var_renge = [0.01 / 360., 1. / 360.]
        var_h = (np.random.rand() * (h_var_renge[1] - h_var_renge[0])) + h_var_renge[0]
        s_base_range = [01. / 100., 30. / 100.]
        base_s = (np.random.rand() * (s_base_range[1] - s_base_range[0])) + s_base_range[0]
        s_var_renge = [0.05 / 100., 1. / 100.]
        var_s = (np.random.rand() * (s_var_renge[1] - s_var_renge[0])) + s_var_renge[0]
        v_var_renge = [0.01 / 100., 1. / 100.]
        v_min_value_range = [0.7, 0.95]
        v_min_value = (np.random.rand(1) * (v_min_value_range[1] - v_min_value_range[0])) + v_min_value_range[0]
        var_v = (np.random.rand() * (v_var_renge[1] - v_var_renge[0])) + v_var_renge[0]
        var = np.array([var_h, var_s, var_v])
        sigma = var ** 0.5
        gauss_s = np.clip(-bg_base+base_s, 0, 1)
        gauss_h = np.ones_like(gauss_s) * base_h
        gauss_v = np.clip(bg_base+v_min_value, 0, 1)
        gauss_hsv = np.stack([gauss_h, gauss_s, gauss_v], axis=2)
        gauss = (hsv2rgb(gauss_hsv) * 255).astype(int)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 2)
        # image[image == 255] = gauss[image == 255]
        return np.clip(gauss - (255 - image), 0, 255)

class ToGray(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if len(sample["img"].shape) > 2:
            if sample["img"].shape[2] > 1:
                do_conv = False
                if sample["img"].dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
                                           np.uint64]:
                    do_conv = True
                    old_type = sample["img"].dtype
                    sample["img"] = sample["img"].astype(np.float32)
                h,w,c = sample["img"].shape
                sample["img"] = np.mean(sample["img"],axis=2).reshape((h,w,1))
                sample["img"] = np.tile(sample["img"], (1,1,3))
                if do_conv:
                    sample["img"] = sample["img"].astype(old_type)
        return sample

class ToBW(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if len(sample["img"].shape) > 2:
            if sample["img"].shape[2] > 1:
                sample["img"] = sample["img"].astype(np.uint8)
                sample["img"] = cv2.cvtColor(sample["img"], cv2.COLOR_RGB2GRAY)
        sample["img"] = cv2.threshold(sample["img"], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #sample["img"] = cv2.adaptiveThreshold(sample["img"], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                                      cv2.THRESH_BINARY, 11, 2)
        if len(sample["img"].shape) == 2:
            sample["img"] = np.tile(np.expand_dims(sample["img"],2), (1,1,3))
        return sample

class Normalize(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        if sample["img"].dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            sample["img"] = sample["img"].astype(np.float32) / 255.
        elif sample["img"].max > 1.1:
            print("Warning: sample image type is float, but values are greater than one.")
            sample["img"] = sample["img"] / 255.
        return sample

def vis_elastic(image, out_path, base_name):
    elastic_alpha = [5.,10.,20.,30.,40.]
    elastic_sigma = [1.,2.,3.,4.]

    for alpha in elastic_alpha:
        for sigma in elastic_sigma:
            sample = {"img": np.copy(image)}
            elastic = OnlyElastic(elastic_alpha=alpha, elastic_sigma=sigma)

            out_image = elastic(sample)["img"]
            cv2.imwrite(join(out_path,'alpha_{}_sigma_{}_'.format(alpha,sigma)+base_name),out_image)
