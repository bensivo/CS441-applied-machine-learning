import numpy as np

def cc_stdev(train_features, train_labels):
    """
    Calculates the conditional class stdev for each class/label combination, ignoring missing values

    Params:
        train_features: np.array [N, X] where N is the number of samples and X is the number of features
        train_labels: np.array [N,] where N is the number of samples.

    Returns:
        np.array [X, Y] where X is the number of features and Y is the number of classes, and each value (x,y) is the stdev of feature x given class y
    """
    num_samples, num_features = train_features.shape
    num_labels = 2
    cc_std = np.zeros((num_features, num_labels))

    for label_ind in range(0, num_labels):
        features = train_features[train_labels == label_ind] # NOTE: here, we're assuming the label indexes are the same as their values.
        for feature_ind in range(0, num_features):
            feature_mean_for_label = np.nanstd(features[:, feature_ind]) # NOTE: the np.mean() function ignores np.NaN values, which we filled in earlier
            cc_std[feature_ind, label_ind] = feature_mean_for_label

    return cc_std
