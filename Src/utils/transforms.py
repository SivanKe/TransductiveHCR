import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import os

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)




if __name__ == '__main__':
    test_im_path = os.path.expandvars('$tibetan_dir/SynthT/18_03_18/one_line_long_text_with_spaces/Images/Qomolangma-Drutsa_Qomolangma-Betsu/1-1/0/0_Qomolangma-Betsu.png')
    save_path = os.path.expandvars('$tibetan_dir/tmp/try_aug/sin_and_elastic.png')
    im = cv2.imread(test_im_path)
    im = elastic_and_sin_transform(im, elastic_alpha=34, elastic_sigma=4, sin_magnitude=4)
    cv2.imwrite(save_path, im)