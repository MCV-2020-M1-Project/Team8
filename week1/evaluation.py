import numpy as np


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def pre_rec_f1(gt, mask):
    """
    Compute precision, recall and F1-score of the mask compared to the ground truth.
    params:
        mask : mask to evaluate
        gt : ground truth
    returns:
        metrics : a tuple containing the metrics (precision, recall, f1).
    """

    if len(gt.shape) < 2 or len(mask.shape) < 2:
        print("ERROR: gt or mask is not matrix.")
        return
    if len(gt.shape) > 2:  # convert to one channel
        gt = gt[:, :, 0]
    if len(mask.shape) > 2:  # convert to one channel
        mask = mask[:, :, 0]
    if gt.shape != mask.shape:
        print("ERROR: The shapes of gt and mask are different.")
        return

    gt_binary = np.where(gt > 128, 1, 0)
    mask_binary = np.where(mask > 128, 1, 0)

    tp = np.sum(mask_binary[(gt_binary == 1) & (mask_binary == 1)])
    fp = np.sum(mask_binary[(gt_binary == 0) & (mask_binary == 1)])
    tn = np.sum(mask_binary[(gt_binary == 0) & (mask_binary == 0)])
    fn = np.sum(mask_binary[(gt_binary == 1) & (mask_binary == 0)])

    f1 = 0.0
    pre = tp / ((tp + fp) * 1.0)
    rec = tp / ((tp + fn) * 1.0)
    if pre != 0.0 and rec != 0.0:
        f1 = (2 * pre * rec) / (pre + rec)

    return pre, rec, f1


if __name__ == '__main__':
    actual = [23]
    predicted = [45, 45, 31, 23, 25]
    print(apk(actual, predicted))
