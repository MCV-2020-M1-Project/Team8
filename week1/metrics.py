import cv2 as cv
import numpy as np


def cosine(x, y):
    """
    Compare histograms based on cosine similarity
    """
    return 1 - np.dot(x, y)/(np.linalg.norm(np.array(x)) * np.linalg.norm(np.array(y)))


def euclidean(x, y):
    """
    Compare histograms based on euclidean distance
    """
    return np.linalg.norm(np.array(x) - np.array(y))


def hellinger_kernel(x, y):
    """
    Compare histograms based on the Hellinger kernel
    """
    return cv.compareHist(x, y, cv.HISTCMP_HELLINGER)


def chi_square(x, y):
    """
    Compare histograms based on Chi-Square
    """
    return cv.compareHist(x, y, cv.HISTCMP_CHISQR)


def correlation(x, y):
    """
    Compare histograms by correlation
    """
    return 1 - cv.compareHist(x, y, cv.HISTCMP_CORREL)


def intersection(x, y):
    """
    Compare histograms by correlation
    """
    return -cv.compareHist(x, y, cv.HISTCMP_INTERSECT)


METRICS = {
    "cosine": cosine,
    "euclidean": euclidean,
    "hellinger_kernel": hellinger_kernel,
    "chi_square": chi_square,
    "correlation": correlation,
    "intersection": intersection
}


def compute_distance(name, x, y):
    return METRICS[name](x, y)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    metric = "chi_square"
    print('try {} metric: distance = {}'.format(metric,
                                                 compute_distance(metric, np.array([1, 2, 3, 4, 5, 6]), np.array([6, 5, 4, 3, 2, 1]))))
