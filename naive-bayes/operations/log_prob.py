import numpy as np

from operations.cc_mean import cc_mean
from operations.cc_stdev import cc_stdev
from operations.log_prior import log_prior

def log_prob(train_features, mu_y, sigma_y, log_py):
    """
    Calculate p(y|x), for each sample in train_features, using bayesian priors

    Params:
        samples: np.array [N, X] where N is the number of samples and X is the number of features
        feature_means: np.array [N,], representing the mean of the PDF used for each feature
        feature_stdevs: np.array [N,], representing the stdev of the PDF used for each feature
        log_prior: np.array [2,], representing the log probabilities for each class in Y

    Returns:
        log_prob: np.array [N, Y], where log_prob[n,y] is roughly p(y|x), using naive bayes
    """

    # Rename the input params to something more readable. We have to keep that signature though, for the assignment
    samples = train_features
    feature_means = mu_y
    feature_stdevs = sigma_y
    log_priors = log_py

    num_samples, _ = samples.shape
    log_prob = np.zeros((num_samples, 2))

    for sample_ind in range(0, num_samples):
        features = samples[sample_ind]

        for label_ind in range(0,2):
            stdevs = feature_stdevs[:, label_ind]
            means = feature_means[:, label_ind]

            # Calculate logp(xi | y=j), for each feature
            feature_log_priors = -0.5 * np.log(2 * np.pi * stdevs**2) - ((features - means)**2 / (2 * stdevs**2))

            # Fetch pre-computed prior, logp(y=j)
            log_prior = log_priors[label_ind]
            
            # Save log prob for this sample, label
            #   p(y=j | X) = logp(y=j) + sum_i(logp(xi | y=j))
            log_prob[sample_ind, label_ind] = log_prior + np.sum(feature_log_priors)

    return log_prob


if __name__ == '__main__':
    # Performing sanity checks on your implementation
    some_feats = np.array([[  1. ,  85. ,  66. ,  29. ,   0. ,  26.6,   0.4,  31. ],
                           [  8. , 183. ,  64. ,   0. ,   0. ,  23.3,   0.7,  32. ],
                           [  1. ,  89. ,  66. ,  23. ,  94. ,  28.1,   0.2,  21. ],
                           [  0. , 137. ,  40. ,  35. , 168. ,  43.1,   2.3,  33. ],
                           [  5. , 116. ,  74. ,   0. ,   0. ,  25.6,   0.2,  30. ]])
    some_labels = np.array([0, 1, 0, 1, 0])

    some_mu_y = cc_mean(some_feats, some_labels)
    some_std_y = cc_stdev(some_feats, some_labels)
    some_log_py = log_prior(some_labels)

    some_log_p_x_y = log_prob(some_feats, some_mu_y, some_std_y, some_log_py)

    assert np.array_equal(some_log_p_x_y.round(3), np.array([[ -20.822,  -36.606],
                                                             [ -60.879,  -27.944],
                                                             [ -21.774, -295.68 ],
                                                             [-417.359,  -27.944],
                                                             [ -23.2  ,  -42.6  ]]))
    print('Passed!')
