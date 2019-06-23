import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math, os
from pathlib import Path
from skimage import data
from skimage.filters import threshold_otsu, gaussian, threshold_adaptive
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, rectangle, erosion, opening
from skimage.color import label2rgb
from skimage.io import  imread, imsave
from skimage.transform import radon, rotate
from scipy.ndimage.measurements import mean, maximum_position
from scipy.signal import find_peaks_cwt, convolve2d
from skimage.color import rgb2grey
from detect_peaks import detect_peaks
import pickle as pkl
import numpy as np
import glob
import cv2


def smooth(a,WSZ=5):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


def point2line(pt, v1_v2):
    v1_v2 = v1_v2.reshape((-1,))
    v1 = v1_v2[0:2]
    v2 = v1_v2[2:]
    d = np.abs(np.linalg.det(np.stack((v1-pt, v2 - v1),0))) / np.linalg.norm(v2 - v1)
    return d


def is_line_in_bb(bb_p1, bb_p2, v1_v2):
    v1_v2 = v1_v2.reshape((-1,))
    v1 = v1_v2[0:2]
    v2 = v1_v2[2:]
    does_cross = np.sign(np.linalg.det(np.stack((bb_p1 - v1, v2 - v1),0))) * np.sign(np.linalg.det(np.stack((bb_p2 - v1, v2 - v1),0))) < 0
    return does_cross


def find_lines_theta_rho(im, tmp_workplace=None, verbose=False, eps=20, max_theta_diff=1.5):
    # find lines using radon transform
    # cleared = (1 - cleared)
    thetas = np.arange(0.0, 180.0, 0.2)
    sinogram = radon(im, theta=thetas, circle=True)
    if verbose:
        plt.imshow(sinogram, aspect='auto')
        plt.savefig(os.path.join(tmp_workplace, 'radon_transform.png'))
        plt.close()
    peak_theta_pos = np.argmax(np.sum(np.square(sinogram), 0))
    # compare peak finding methods from https://github.com/MonsieurV/py-findpeaks
    rhos = sinogram[:, peak_theta_pos]
    smoothed_rhos = smooth(rhos, 11)
    if verbose:
        f, ax = plt.subplots(1)
        ax.plot(np.arange(len(rhos)), rhos)
        plt.savefig(os.path.join(tmp_workplace, 'rhos.png'))
        plt.close()
        ax.plot(np.arange(len(rhos)), rhos)
    rhos = smoothed_rhos

    peak_rhos = detect_peaks(rhos, valley=False).reshape((-1,))

    peak_rhos[::-1].sort()
    peak_rhos = peak_rhos[peak_rhos > 6]
    peak_rhos = peak_rhos[peak_rhos <len(rhos)]
    mean_rho_dist = np.mean(-peak_rhos[1:]+peak_rhos[:-1])
    # find better approximation to theta and rho in each peak
    peak_thetas = np.zeros(peak_rhos.shape)
    for i, rho in enumerate(peak_rhos):
        sinw, sinh = sinogram.shape
        surrounding_im = sinogram[max(rho - eps,0):min(rho + eps + 1,sinw),
                         max(peak_theta_pos - eps,0):min(peak_theta_pos + eps + 1,sinh)]
        surrounding_im = gaussian(surrounding_im, sigma=2)
        cur_rho, cur_theta = np.unravel_index(np.argmax(surrounding_im), surrounding_im.shape)

        max_theta_diff = max_theta_diff
        while np.abs(thetas[peak_theta_pos - eps + cur_theta] - thetas[peak_theta_pos]) > max_theta_diff:
            surrounding_im[cur_rho, cur_theta] = 0
            cur_rho, cur_theta = np.unravel_index(np.argmax(surrounding_im), surrounding_im.shape)
        peak_thetas[i] = thetas[peak_theta_pos - eps + cur_theta]
        peak_rhos[i] = rho - eps + cur_rho

    peak_thetas = [theta-90 for theta in peak_thetas]
    return peak_thetas, peak_rhos, mean_rho_dist


def im_lines_from_theta_rho(img, peak_thetas, peak_rhos):
    # calculate lines by radon peaks
    rad_tetas = [np.radians(theta) for theta in peak_thetas]
    x_center = math.floor(img.shape[1] / 2)
    y_center = math.floor(img.shape[0] / 2)
    lines_x1_y1_x2_y2 = np.zeros((len(peak_rhos), 4))
    for p in range(len(peak_rhos)):
        cur_rho = peak_rhos[p]

        cur_line = np.array([[y_center - cur_rho, -x_center], [y_center - cur_rho, x_center]]).T
        # Rotate
        rotation_mat = np.array([[np.cos(rad_tetas[p]), -np.sin(rad_tetas[p])],
                                 [np.sin(rad_tetas[p]), np.cos(rad_tetas[p])]])
        rotated_lines = np.matmul(rotation_mat, cur_line)
        # Shift by image center
        rotated_lines[0, :] = rotated_lines[0, :] + y_center
        rotated_lines[1, :] = rotated_lines[1, :] + x_center
        # add line to peak lines
        lines_x1_y1_x2_y2[p, :] = [rotated_lines[0, 0], rotated_lines[1, 0], rotated_lines[0, 1], rotated_lines[1, 1]]

    lines_x1_y1_x2_y2 = np.split(lines_x1_y1_x2_y2, lines_x1_y1_x2_y2.shape[0], axis=0)
    return lines_x1_y1_x2_y2


def separate_ccs_by_lines(regions, label_image, lines_x1_y1_x2_y2, mean_rho_dist):
    line2comps = {}
    line2centroid = {}
    line2im = {}
    for region in regions:
        centriod = region.centroid
        bb = region.bbox
        bb_p1 = bb[0:2]
        bb_p2 = bb[2:]
        line_in_p_bb = lambda x: is_line_in_bb(bb_p1, bb_p2, x)
        get_dist_from_p = lambda x: point2line(centriod, x)
        min_d = None
        argmin_d = None
        num_crossing_lines = 0
        for line_i, line in enumerate(lines_x1_y1_x2_y2):
            if line_in_p_bb(line):
                num_crossing_lines += 1
            cur_dist = get_dist_from_p(line)
            min_d = min(min_d, cur_dist) if min_d is not None else cur_dist
            argmin_d = line_i if (cur_dist == min_d) else argmin_d
        if (num_crossing_lines > 1) or (min_d > 1.1 * mean_rho_dist):
            continue
        line2comps[argmin_d] = line2comps.setdefault(argmin_d, [])
        line2comps[argmin_d].append(region.label)
        line2centroid[argmin_d] = line2centroid.setdefault(argmin_d, [])
        line2centroid[argmin_d].append(region.centroid)
        line2im[argmin_d] = line2im.setdefault(argmin_d, np.zeros(label_image.shape))+ \
                            (label_image == region.label).astype(int)
    return line2im, line2centroid

def rotate_crop_images(orig_im, line2im, peak_thetas, peak_rhos, thresh_area):
    output_line2im = {}
    items_list = list(line2im.items())
    for i, (line, im) in enumerate(items_list):
        line_image = rotate(im, (-peak_thetas[i]), center=[peak_rhos[i], (im.shape[1] / 2)])
        labels = label(line_image, neighbors=4, background=0)
        regions = regionprops(labels)
        if len([region for region in regions if region.area >= thresh_area]) > 0:
            min_rows = int(np.percentile(np.array([region.bbox[0] for region in regions if region.area >= thresh_area]), 2))
            max_rows = int(
                np.percentile(np.array([region.bbox[2] for region in regions if region.area >= thresh_area]), 98))
        else:
            min_rows = int(
                np.percentile(np.array([region.bbox[0] for region in regions]), 1))
            max_rows = int(
                np.percentile(np.array([region.bbox[2] for region in regions]), 98))
        new_im = np.copy(orig_im)
        if i < (len(line2im.items())-1):
            next_image = items_list[i+1][1]
            new_im[next_image == 1] = 255
        if i > 0:
            prev_im = items_list[i-1][1]
            new_im[prev_im == 1] = 255

        new_im = rotate(new_im, (-peak_thetas[i]), center=[peak_rhos[i], (im.shape[1] / 2)])
        if len(orig_im.shape) == 3:
            # CHANGED added 3 and -3
            new_im = new_im[max(0, min_rows-3):min(line_image.shape[0], max_rows+3), :, :]
        else:
            # CHANGED added 3 and -3
            new_im = new_im[max(0, min_rows-3):min(line_image.shape[0], max_rows+3), :]
        output_line2im[line] = new_im
    return output_line2im


def clean_image(bw, threshold=0.1):
    new_bw = np.zeros(bw.shape)
    label_image = label(bw)
    regions = regionprops(label_image)
    thresh_area = threshold * np.percentile([reg.area for reg in regions], 90)
    for reg in regionprops(label(bw)):
        if reg.area >= thresh_area:
            new_bw = new_bw + (label_image == reg.label).astype(int)
    bw = new_bw.astype(int)
    return bw

# CHANGED eps from 10 to 20
# CHANGED max teta diff to 1.0
def im2lines(img_path, tmp_workplace=None, verbose=False,
             addaptive=False, eps=20, max_theta_diff=1.0, do_morphologic_cleaning=True):
    orig_image = imread(img_path)
    if len(orig_image.shape) > 2 and  orig_image.shape[2] > 1:
        image = rgb2grey(orig_image)
    else:
        image = orig_image
    # apply threshold
    if addaptive:
        block_size = 35
        bw = threshold_adaptive(image, block_size, offset=0.1)
    else:
        thresh = threshold_otsu(image) # Fisher Discriminant Analysis backround intensity detector
        bw = image > thresh
    bw = 1 - bw.astype(int)
    # remove artifacts connected to image border
    if do_morphologic_cleaning:
        cleared = clean_image(bw, threshold=0.1)
        cleared = closing(cleared, square(10))#square(7)) # for drutsa - 5
        cleared = clear_border(cleared)
        cleared = opening(cleared, rectangle(width=2, height=40))#rectangle(width=4, height=28))
    else:
        cleared=bw
    peak_thetas, peak_rhos, mean_rho_dist = find_lines_theta_rho(cleared, tmp_workplace=tmp_workplace,
                                                                 verbose=verbose, eps=eps,
                                                                 max_theta_diff=max_theta_diff)
    lines_x1_y1_x2_y2 = im_lines_from_theta_rho(cleared, peak_thetas, peak_rhos)
    label_image = label(bw, neighbors=4, background=0)
    regions = regionprops(label_image)
    thresh_area = 0.1 * np.percentile([reg.area for reg in regions], 80)
    line2im, line2centroid = separate_ccs_by_lines(regions, label_image, lines_x1_y1_x2_y2, mean_rho_dist)
    line2im = rotate_crop_images(orig_image, line2im, peak_thetas, peak_rhos, thresh_area)
    return line2im






