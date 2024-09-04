import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from operations.split_df_train_eval import split_df_train_eval
from operations.split_df_feature_label import split_df_feature_label
from operations.cc_mean import cc_mean
from operations.cc_stdev import cc_stdev
from operations.log_prior import log_prior
from operations.log_prob import log_prob

df = pd.read_csv('diabetes.csv')

# Split data into training and evaluation sets, features and labels
train_df, eval_df = split_df_train_eval(df = df, train_percentage = 80, seed = 12345)
train_features, train_labels = split_df_feature_label(df=train_df, label_col='Outcome')
eval_features, eval_labels = split_df_feature_label(df=eval_df, label_col='Outcome')
print(f'Split data into {train_features.shape[0]} training samples and {eval_features.shape[0]} evaluation samples')

# Pre-process data
# Some of the columns exhibit missing values. We will use a Naive Bayes Classifier later that will treat such missing values in a special way. 
# To be specific, for attribute 3 (Diastolic blood pressure), attribute 4 (Triceps skin fold thickness), attribute 6 (Body mass index), and attribute 8 (Age), we should regard a value of 0 as a missing value.
train_df_with_nans = train_df.copy(deep=True)
eval_df_with_nans = eval_df.copy(deep=True)
for col in ['BloodPressure', 'SkinThickness', 'BMI', 'Age']:
    train_df_with_nans[col] = train_df_with_nans[col].replace(0, np.nan)
    eval_df_with_nans[col] = eval_df_with_nans[col].replace(0, np.nan)
train_features = train_df_with_nans.loc[:, train_df_with_nans.columns != 'Outcome'].values
eval_features = eval_df_with_nans.loc[:, eval_df_with_nans.columns != 'Outcome'].values


# # Running the classifier step-by-step
#
# train_log_prior = log_prior(train_labels)
# train_cc_mean = cc_mean(train_features, train_labels)
# train_cc_stdev = cc_stdev(train_features, train_labels)
# log_probs = log_prob(eval_features, train_cc_mean, train_cc_stdev, train_log_prior)

# print('log(p(y)):', train_log_prior)
# print('Conditional class means', train_cc_mean)
# print('Conditional class stddevs', train_cc_stdev)
# print('log probs:', log_probs)

class NBClassifier():
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.log_py = log_prior(train_labels)
        self.mu_y = self.get_cc_means()
        self.sigma_y = self.get_cc_std()
        
    def get_cc_means(self):
        mu_y = cc_mean(self.train_features, self.train_labels)
        return mu_y
    
    def get_cc_std(self):
        sigma_y = cc_stdev(self.train_features, self.train_labels)
        return sigma_y
    
    def predict(self, features):
        log_p_x_y = log_prob(features, self.mu_y, self.sigma_y, self.log_py)
        return log_p_x_y.argmax(axis=1)

diabetes_classifier = NBClassifier(train_features, train_labels)
train_pred = diabetes_classifier.predict(train_features)
eval_pred = diabetes_classifier.predict(eval_features)

train_acc = (train_pred==train_labels).mean()
eval_acc = (eval_pred==eval_labels).mean()
print(f'The training data accuracy of your trained model is {train_acc}')
print(f'The evaluation data accuracy of your trained model is {eval_acc}')

