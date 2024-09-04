import pandas as pd

def split_df_feature_label(df, label_col):
    """
    Given a dataframe containing both the input features and the labels,
    split it into 2 np arrays, one for the input features, and one for the class labels

    Params:
        df: pd.DataFrame, the dataframe to split
        label_col: str, the name of the column containing the labels

    Returns:
        features: np.array [N, X], where N is the number of samples and X is the number of features
        labels: np.array [N,], where N is the number of samples
    """
    # Split the train df into input-features (everythign but 'Outcome') and labels ('Outcome')
    features = df.loc[:, df.columns != 'Outcome'].values # Split the training df into 2 np arrays, one for the input features, and one for the class labels
    labels = df.loc[:, 'Outcome'].values
        # .loc[<rows> , <cols>] is pd's way of making slices from a df
        # .values converts the results to a np array

    return features, labels
