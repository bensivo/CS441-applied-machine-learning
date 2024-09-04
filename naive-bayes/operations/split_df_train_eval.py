import numpy as np
import pandas as pd


def split_df_train_eval(df, train_percentage=80, seed=None):
    """
    Split a dataframe into training and evaluation sets, using the given percentage for training

    Params:
        df: pd.DataFrame
        train_percentage: int, the percentage of the data to use for training (out of 100)
        seed: int, seed for random number generator

    Returns:
        train_df: pd.DataFrame, the training set
        eval_df: pd.DataFrame, the evaluation set
    """
    np_random = np.random.RandomState(seed=seed)
    rand_unifs = np_random.uniform(low=0, high=1,size=df.shape[0]) # Random numbers between 0 and 1
    division_thresh = np.percentile(rand_unifs, 80) # Calculate the threshold that puts 80% of the samples below, and 20% above 
    train_indicator = rand_unifs < division_thresh  # boolean array, whether this sample is in the training set
    eval_indicator = rand_unifs >= division_thresh  # boolean array, whether this sample is in the evaluation set

    # Create DF's for the training and evaluation datasets
    train_df = df[train_indicator].reset_index(drop=True)
    eval_df = df[eval_indicator].reset_index(drop=True)
        # .reset_index(drop=True) re-initializes the default numerical indexes used to access elements in this df. 
        #   Otherwise, it keesp teh indexes from the original

    return train_df, eval_df

