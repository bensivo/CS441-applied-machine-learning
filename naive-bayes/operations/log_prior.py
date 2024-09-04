import numpy as np

def log_prior(train_labels):
    """
    Given an array of training labels, calculate the log prior probabilities of each class

    Note:
        This implementation assumes the labels are binary (0, 1)

    Returns:
        log_prior: np.array [log(p(y=0)), log(p(y=1))]
    """
    num_0 = np.count_nonzero(train_labels == 0)
    num_1 = np.count_nonzero(train_labels == 1)

    prob = np.zeros((2, 1))
    prob[0] = num_0 / len(train_labels)
    prob[1] = num_1 / len(train_labels)

    log_prob = np.log(prob)
    assert log_prob.shape == (2,1)
    return log_prob

if __name__ == '__main__':
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    log_py = log_prior(labels)
    assert np.array_equal(log_py.round(3), np.array([[-0.916], [-0.511]]))
    print('Passed!')
