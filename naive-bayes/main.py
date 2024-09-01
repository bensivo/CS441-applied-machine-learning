import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read csv
df = pd.read_csv('diabetes.csv')

# Split the dataset
np_random = np.random.RandomState(seed=12345)
rand_unifs = np_random.uniform(low=0, high=1,size=df.shape[0]) # Random numbers between 0 and 1
division_thresh = np.percentile(rand_unifs, 80) # Calculate the threshold that puts 80% of the samples below, and 20% above 
train_indicator = rand_unifs < division_thresh  # boolean array, whether this sample is in the training set
eval_indicator = rand_unifs >= division_thresh  # boolean array, whether this sample is in the evaluation set


# Create DF's for the training and evaluation datasets
train_df = df[train_indicator].reset_index(drop=True)
eval_df = df[eval_indicator].reset_index(drop=True)
    # .reset_index(drop=True) re-initializes the default numerical indexes used to access elements in this df. Otherwise, it keesp teh indexes from the original

# Split the train df into input-features (everythign but 'Outcome') and labels ('Outcome')
train_features = train_df.loc[:, train_df.columns != 'Outcome'].values # Split the training df into 2 np arrays, one for the input features, and one for the class labels
train_labels = train_df.loc[:, 'Outcome'].values
    # .loc[<rows> , <cols>] is pd's way of making slices from a df
    # .values converts the results to a np array

# Split the eval df into input-features and labels (the 'Outcome' column
eval_features = eval_df.loc[:, eval_df.columns != 'Outcome'].values
eval_labels = eval_df.loc[:, 'Outcome'].values

print('Data Split shape', train_features.shape, train_labels.shape, eval_features.shape, eval_labels.shape)


# Pre-process data
# Some of the columns exhibit missing values. We will use a Naive Bayes Classifier later that will treat such missing values in a special way. 
# To be specific, for attribute 3 (Diastolic blood pressure), attribute 4 (Triceps skin fold thickness), attribute 6 (Body mass index), and attribute 8 (Age), we should regard a value of 0 as a missing value.
train_df_with_nans = train_df.copy(deep=True)
eval_df_with_nans = eval_df.copy(deep=True)
for col in ['BloodPressure', 'SkinThickness', 'BMI', 'Age']:
    train_df_with_nans[col] = train_df_with_nans[col].replace(0, np.nan)
    eval_df_with_nans[col] = eval_df_with_nans[col].replace(0, np.nan)
train_features_with_nans = train_df_with_nans.loc[:, train_df_with_nans.columns != 'Outcome'].values
eval_features_with_nans = eval_df_with_nans.loc[:, eval_df_with_nans.columns != 'Outcome'].values



def log_prior(train_labels):
    """
    Calculates our bayesian prior for training labels, log prob of y=0, and y=1

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

log_prior_train_labels = log_prior(train_labels)
print('log(p(y)):', log_prior_train_labels)


def cc_mean_ignore_missing(train_features, train_labels):
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
            feature_mean_for_label = np.mean(features[:, feature_ind]) # NOTE: the np.mean() function ignores np.NaN values, which we filled in earlier
            cc_mean[feature_ind, label_ind] = feature_mean_for_label

    return cc_mean

cc_mean_train = cc_mean_ignore_missing(train_features, train_labels)
print('Conditional class means', cc_mean_train)


def cc_stddev_ignore_missing(train_features, train_labels):
    """
    Calculates the conditional class stddev for each class/label combination, ignoring missing values

    Params:
        train_features: np.array [N, X] where N is the number of samples and X is the number of features
        train_labels: np.array [N,] where N is the number of samples.

    Returns:
        np.array [X, Y] where X is the number of features and Y is the number of classes, and each value (x,y) is the stddev of feature x given class y
    """
    num_samples, num_features = train_features.shape
    num_labels = 2
    cc_std = np.zeros((num_features, num_labels))

    for label_ind in range(0, num_labels):
        features = train_features[train_labels == label_ind] # NOTE: here, we're assuming the label indexes are the same as their values.
        for feature_ind in range(0, num_features):
            feature_mean_for_label = np.std(features[:, feature_ind]) # NOTE: the np.mean() function ignores np.NaN values, which we filled in earlier
            cc_std[feature_ind, label_ind] = feature_mean_for_label

    return cc_std

cc_stddev_train = cc_stddev_ignore_missing(train_features, train_labels)
print('Conditional class stddevs', cc_stddev_train)


def log_prob(train_features, mu_y, sigma_y, log_py):
    """
    Calculate p(y|x), for each sample in train_features, using bayesian priors

    Params:
        train_features: np.array [N, X] where N is the number of samples and X is the number of features
        mu_y: Gaussian constant for mean of the PDF used on Y
        sigma_y: Gaussian constant for the spread of the PDF used on Y
        log_py: Prior. Log probabilities for each class in Y

    Returns:
        log_prob: np.array [N, Y], where log_prob[n,y] is roughly p(y|x), using naive bayes
    """
    num_samples, num_features = train_features.shape
    output = np.zeros((num_samples, 2))
    
    return output
