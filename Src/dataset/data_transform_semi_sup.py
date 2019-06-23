import numpy as np
import cv2
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.color import hsv2rgb
import scipy

class ElasticAndSineDouble(object):

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
        sample["img1"] = self.__elastic_and_sin_transform(sample["img1"])
        sample["img2"] = self.__elastic_and_sin_transform(sample["img2"])
        return sample


class ResizeDouble(object):
    def __init__(self, hight=64):
        self.hight = hight

    def __call__(self, sample):

        def get_im_width(img, hight):
            aspect_ratio = hight / img.shape[0]
            return int(aspect_ratio * img.shape[1])

        def should_resize(img, width, hight):
            return ((img.shape[0] != hight) or (img.shape[1] != width))

        images = [sample["img1"], sample["img2"]]
        widths = [get_im_width(img, self.hight) for img in images]
        max_width = max(widths)
        resized_images = [scipy.misc.imresize(img, (self.hight, max_width), interp='lanczos') for img in images if
                          should_resize(img, max_width, self.hight)]
        sample["img1"] = resized_images[0]
        sample["img2"] = resized_images[1]
        return sample

class AddWidthDouble(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample["img_width1"] = sample["img1"].shape[1]
        sample["img_width2"] = sample["img2"].shape[1]
        return sample

class RotationDouble(object):
    def __init__(self, angle=20, fill_value=0, p = 0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def _rotate_im(self, img):
        if np.random.uniform(0.0, 1.0) > self.p:
            return img
        h,w,_ = img.shape
        ang_rot = np.random.uniform(self.angle) - self.angle/2
        transform = cv2.getRotationMatrix2D((w/2, h/2), ang_rot, 1)
        img = cv2.warpAffine(img, transform, (w,h), borderValue = self.fill_value)
        return img

    def __call__(self, sample):
        sample["img1"] = self._rotate_im(sample["img1"])
        sample["img2"] = self._rotate_im(sample["img2"])
        return sample

class ColorGradGausNoiseDouble(object):

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, sample):
        sample["img1"] = self.__color_aug(sample["img1"], self.p)
        sample["img2"] = self.__color_aug(sample["img2"], self.p)
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
    def __color_aug(image, p):
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

        if np.random.uniform(0.0, 1.0) > p:
            return image

        bg_grad_axis = np.random.randint(2)
        sin_start_range = [0, np.random.rand(1) * np.pi]
        sin_size_range = [0.02, 2 * np.pi]

        bg_base = ColorGradGausNoiseDouble.__gradient_background(image, sin_start_range, sin_size_range, 0, axis=bg_grad_axis)
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

class NormalizeDouble(object):

    def __init__(self):
        pass

    def _norm(self, img):
        if img.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            img = img.astype(np.float32) / 255.
        elif img.max > 1.1:
            print("Warning: sample image type is float, but values are greater than one.")
            img = img / 255.
        return img

    def __call__(self, sample):
        sample["img1"] = self._norm(sample["img1"])
        sample["img2"] = self._norm(sample["img2"])
        return sample
