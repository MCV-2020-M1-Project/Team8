import os
import pickle
import numpy as np
from histograms import compute_descriptor
from metrics import compute_distance


def compute_histograms(db, dataset_dir, database_dir):
    """
    Compute histograms of all images in the dataset .
    Result will be saved for further usage.

    Note: color_space represents hist dataset file as well
    params:
        db: specified descriptor
        dataset_dir: dataset directory
    return:
        histograms: A list of histograms.
    """

    histograms = []
    if os.path.exists(dataset_dir):
        img_files = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith(".jpg")]
        print("Total images: {}".format(len(img_files)))
        count = 0
        for img_file in sorted(img_files):
            histograms.append(compute_descriptor(db, img_file))
            if count % 100 == 0 or count == len(img_files) - 1:
                print("Processed image {} ".format(count))
            count += 1
        with open(database_dir + '/' + db + '.pkl', 'wb') as f:
            pickle.dump(histograms, f)
    else:
        print("Directory does not exist")
    return histograms


def load_database(db, dataset_dir, database_dir):
    histograms = []
    if os.path.exists(database_dir + '/' + db + '.pkl'):
        with open(database_dir + '/' + db + '.pkl', 'rb') as f:
            histograms = pickle.load(f)
    else:
        print("No such file")
    if len(histograms) == 0:
        histograms = compute_histograms(db, dataset_dir, database_dir)
    return np.asarray(histograms)


def load_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret


def save_pickle(path, var):
    with open(path, 'wb') as f:
        pickle.dump(var, f)


def k_nearest_search(hists, query, ground_truth, metric="euclidean_distance", k=10):
    """
    Choose the top K results that closest to the query using different metrics.
    params:
        hists: database histogram
        query: query histogram
        ground_truth: ground truth corresponding to query
        metric: metric specified
        k: number of top result retrieved
    return:
        dist_to_img: A list of K elements.
    """
    if k > len(hists):
        return "K is larger than proper length"

    dist_to_img = []
    distance_gt = 0
    for idx, hist in enumerate(hists):
        # calculate distance to each histogram
        distance = compute_distance(metric, hist, query)

        # save the distance of ground truth picture
        if idx == ground_truth:
            distance_gt = distance

        # sort out the best results by descending order
        if len(dist_to_img) < k:
            dist_to_img.append([distance, idx])
            dist_to_img = sorted(dist_to_img)
        else:
            if distance < dist_to_img[-1][0]:
                dist_to_img[-1] = [distance, idx]
                dist_to_img = sorted(dist_to_img)

    # prioritize gt picture if distances are equals
    #     if (not reverse and distance_gt <= dist_to_img[-1][0]) or (reverse and distance_gt >= dist_to_img[-1][0]):
    #         dist_to_img[-1] = [distance_gt, ground_truth]

    #     dist_to_img = sorted(dist_to_img, reverse = reverse)
    #     ## find the rank of gt picture
    #     rank = 1
    #     for i, dti in enumerate(dist_to_img):
    #         if dti[-1] == ground_truth:
    #             rank = i + 1
    #             break

    return sorted(dist_to_img)
