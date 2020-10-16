# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import cv2 as cv
import numpy as np


def gray_histogram(path):
    """
    compute histogram of an image in grayscale
    params:
        path: A path that specifies the location of image
    return:
        hist: 1D histogram.
    """
    image = cv.imread(path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv.normalize(hist, hist).flatten()
    return hist


def hsv_1d_histogram(path):
    """
    compute 1D histogram of an image in HSV color space
    params:
        path: A path that specifies the location of image
    return:
        hist: 3D histogram flatted into an 1D array.
    """
    image = cv.imread(path)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    channels = cv.split(hsv)
    hists = []
    # we should focus more on hue channel to distinguish colors
    # lighting/brightness channel is not necessarily divided into too many bins
    for channel, bin_val, max_val in zip(channels, [180, 256, 64], [180, 256, 256]):
        hist = cv.calcHist([channel], [0], None, [bin_val], [0, max_val])
        hist = cv.normalize(hist, hist)
        hists.extend(hist)
    # flatten into a 1D array
    return np.stack(hists).flatten()


def rgb_1d_histogram(path):
    """
    compute 1D histogram of an image in RGB color space
    params:
        path: A path that specifies the location of image
    return:
        hist: 3D histogram flatted into an 1D array.
    """
    image = cv.imread(path)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    channels = cv.split(hsv)
    hists = []
    for channel in channels:
        hist = cv.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv.normalize(hist, hist)
        hists.extend(hist)
    # flatten into a 1D array
    return np.stack(hists).flatten()


def ycrcb_1d_histogram(path):
    """
    compute 1D histogram of an image in YCrCb color space
    params:
        path: A path that specifies the location of image
    return:
        hist: 3D histogram flatted into an 1D array.
    """
    image = cv.imread(path)
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    channels = cv.split(ycrcb)
    hists = []
    for channel in channels:
        hist = cv.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv.normalize(hist, hist)
        hists.extend(hist)
    # flatten into a 1D array
    return np.stack(hists).flatten()


def lab_1d_histogram(path):
    """
    compute 1D histogram of an image in LAB color space
    params:
        path: A path that specifies the location of image
    return:
        hist: 3D histogram flatted into an 1D array.
    """
    image = cv.imread(path)
    lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    channels = cv.split(lab)
    hists = []
    for channel in channels:
        hist = cv.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv.normalize(hist, hist)
        hists.extend(hist)
    # flatten into a 1D array
    return np.stack(hists).flatten()


def hsv_3d_histogram(path):
    """
    compute 3D histogram of an image in HSV color space
    params:
        path: A path that specifies the location of image
    return:
        hist: 3D histogram flatted into an 1D array.
    """
    image = cv.imread(path)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # we should focus more on hue channel to distinguish colors
    # lighting/brightness channel is not necessarily divided into too many bins
    hist = cv.calcHist([hsv], [0, 1, 2], None, [180, 64, 16], [0, 180, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()
    return hist


def rgb_3d_histogram(path):
    """
    compute 3D histogram of an image in RGB color space
    params:
        path: path pointing to the image to compute
    return:
        hist: 3D histogram flatted into an 1D array.
    """
    image = cv.imread(path)
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    hist = cv.calcHist([rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()
    return hist


def ycrcb_3d_histogram(path):
    """
    compute 3D histogram of an image in YCrCb color space
    params:
        path: path pointing to the image to compute
    returns:
        hist: 3D histogram flatted into an 1D array
    """
    image = cv.imread(path)
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    hist = cv.calcHist([ycrcb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist)
    return hist.flatten()


def lab_3d_histogram(path):
    """
    compute 3D histogram of an image in lab color space
    params:
        path: path pointing to the image to compute
    returns:
        hist: 3D histogram flatted into an 1D array
    """
    image = cv.imread(path)
    lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    hist = cv.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist)
    return hist.flatten()


DESCRIPTOR_FUNCTIONS = {
    "gray_histogram": gray_histogram,
    "hsv_1d_histogram": hsv_1d_histogram,
    "rgb_1d_histogram": rgb_1d_histogram,
    "ycrcb_1d_histogram": ycrcb_1d_histogram,
    "lab_1d_histogram": lab_1d_histogram,
    "hsv_3d_histogram": hsv_3d_histogram,
    "rgb_3d_histogram": rgb_3d_histogram,
    "ycrcb_3d_histogram": ycrcb_3d_histogram,
    "lab_3d_histogram": lab_3d_histogram
}


def compute_descriptor(descriptor, path):
    return DESCRIPTOR_FUNCTIONS[descriptor](path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    name = "rgb_3d_histogram"
    print('try {}: shape = {}'.format(name, compute_descriptor(name, os.path.join('images', '00000.jpg')).shape))

