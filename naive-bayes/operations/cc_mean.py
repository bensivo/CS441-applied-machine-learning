import numpy as np

def cc_mean(train_features, train_labels):
    """
    Calculates the conditional class mean for each class/label combination, ignoring missing values

    Params:
        train_features: np.array [N, X] where N is the number of samples and X is the number of features
        train_labels: np.array [N,] where N is the number of samples.

    Returns:
        np.array [X, Y] where X is the number of features and Y is the number of classes, and each value (x,y) is the mean of feature x given class y
    """
    num_samples, num_features = train_features.shape
    num_labels = 2
    cc_mean = np.zeros((num_features, num_labels))

    for label_ind in range(0, num_labels):
        features = train_features[train_labels == label_ind] # NOTE: here, we're assuming the label indexes are the same as their values.
        for feature_ind in range(0, num_features):
            feature_mean_for_label = np.nanmean(features[:, feature_ind]) # NOTE: the np.mean() function ignores np.NaN values, which we filled in earlier
            cc_mean[feature_ind, label_ind] = feature_mean_for_label

    return cc_mean



if __name__ == '__main__':
    features = np.array([[  1. ,  85. ,  66. ,  29. ,   0. ,  26.6,   0.4,  31. ],
                         [  8. , 183. ,  64. ,   0. ,   0. ,  23.3,   0.7,  32. ],
                         [  1. ,  89. ,  66. ,  23. ,  94. ,  28.1,   0.2,  21. ],
                         [  0. , 137. ,  40. ,  35. , 168. ,  43.1,   2.3,  33. ],
                         [  5. , 116. ,  74. ,   0. ,   0. ,  25.6,   0.2,  30. ]])

    labels = np.array([0, 1, 0, 1, 0])

    res = cc_mean_ignore_missing(features, labels)

    assert np.array_equal(res.round(2), np.array([[  2.33,   4.  ],
                                                  [ 96.67, 160.  ],
                                                  [ 68.67,  52.  ],
                                                  [ 17.33,  17.5 ],
                                                  [ 31.33,  84.  ],
                                                  [ 26.77,  33.2 ],
                                                  [  0.27,   1.5 ],
                                                  [ 27.33,  32.5 ]]))
    print('Passed!')
