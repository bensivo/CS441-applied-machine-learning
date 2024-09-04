import pandas as pd
import numpy as np

from operations.split_df_train_eval import split_df_train_eval
from operations.split_df_feature_label import split_df_feature_label
from operations.cc_mean import cc_mean
from operations.cc_stdev import cc_stdev
from operations.log_prior import log_prior
from operations.log_prob import log_prob

df = pd.read_csv('diabetes.csv')

# Split data into training and evaluation sets, then into features and labels
train_df, eval_df = split_df_train_eval(df = df, train_percentage = 80, seed = 12345)
train_features, train_labels = split_df_feature_label(df=train_df, label_col='Outcome')
eval_features, eval_labels = split_df_feature_label(df=eval_df, label_col='Outcome')
print(f'Split data into {train_features.shape[0]} training samples and {eval_features.shape[0]} evaluation samples')

# Replace some 0 values with NaNs
train_df_with_nans = train_df.copy(deep=True)
eval_df_with_nans = eval_df.copy(deep=True)
for col in ['BloodPressure', 'SkinThickness', 'BMI', 'Age']:
    train_df_with_nans[col] = train_df_with_nans[col].replace(0, np.nan)
    eval_df_with_nans[col] = eval_df_with_nans[col].replace(0, np.nan)
train_features = train_df_with_nans.loc[:, train_df_with_nans.columns != 'Outcome'].values
eval_features = eval_df_with_nans.loc[:, eval_df_with_nans.columns != 'Outcome'].values

class NBClassifier():
    def fit(self, features, labels):
        self.logp_y = log_prior(train_labels)
        self.mu_y = cc_mean(features, labels)
        self.sigma_y = cc_stdev(features, labels)
    
    def predict(self, features):
        logp_x_y = log_prob(features, self.mu_y, self.sigma_y, self.logp_y)
        return logp_x_y.argmax(axis=1)

diabetes_classifier = NBClassifier()
diabetes_classifier.fit(train_features, train_labels)

train_pred = diabetes_classifier.predict(train_features)
eval_pred = diabetes_classifier.predict(eval_features)

train_acc = (train_pred==train_labels).mean()
eval_acc = (eval_pred==eval_labels).mean()
print(f'The training data accuracy of your trained model is {train_acc}')
print(f'The evaluation data accuracy of your trained model is {eval_acc}')

