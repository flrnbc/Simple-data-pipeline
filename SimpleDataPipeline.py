import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# DOWNLOAD DATA
def fetch_data(data_url, data_dir, data_name):
    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, data_name + ".tgz")
    # only download
    if not os.path.exists(tgz_path):
        urllib.request.urlretrieve(data_url, tgz_path)  # download tar-file
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=data_dir)  # extract to housing_path
        housing_tgz.close()


# LOAD DATA with pandas
def load_data_pd(data_dir, data_name):
    data_dir = os.path.join(data_dir, data_name + ".csv")
    print("Loaded {}.".format(data_dir))
    return pd.read_csv(data_dir)


# SPLITTING DATA
# split into test and train set using stratified shuffle split


def shuffle_split_data(dataframe, bins, ratio=0.2):
    """
    Splits data into test and training set using StratifiedShuffleSplit (sklearn).

    Input:
    - dataframe: data to be split
    - bins: category used for StratifiedShuffleSplit (use pd.cut)
    - ratio: len(test_set)/len(dataframe)

    Output:
    - Tuple (train_set, test_set).
    """
    shuffle_split = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    for train_index, test_index in shuffle_split.split(dataframe, bins):
        strat_train_set = dataframe.loc[train_index]
        strat_test_set = dataframe.loc[test_index]
    return (strat_train_set, strat_test_set)


# CUSTOM TRANSFORMER(S)
# combine attributes
# TODO: add names for the combined attributes?


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Transformer to add combined attributes.

    Class attributes:
    - combine_attrs: list of tuples (m, n). These will combine the m-th and n-th
      column/attributes ('divide m-th by n-th column')

    Input for transform:
    - X (np) array (only numerical entries)

    Output:
    - X with the appended attributes.
    """

    def __init__(self, combine_attrs):
        # no *args, **kargs (BaseEstimator requires explicit keyword args
        # (design choice?)
        self.combine_attrs = combine_attrs

    def fit(self, X, y=None):
        return self

    # why not work with pd dataframes?
    def transform(self, X, y=None):
        combined_columns = []
        for pair in self.combine_attrs:
            attr = X[:, pair[0]] / X[:, pair[1]]
            combined_columns.append(attr)
        for col in combined_columns:
            X = np.c_[X, attr]
        return X


# DATA PIPELINE


def num_pipeline(combine_attrs):
    """
    Simple pipeline to transform numerical data using sklearn's Pipeline.

    Executes the following transformations:
    - 'imputer': Replaces missing values in a column by corresponding
                 median values.
    - 'attribs_adder': Applies the CombinedAttributesAdder with
                       parameters combine_attrs.
    - 'std_scaler': Scales the values.
    """
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder(combine_attrs)),
            ("std_scaler", StandardScaler()),
        ]
    )
    return num_pipeline


def full_pipeline(df, combine_attrs):
    """
    Input:
    - num_data: typically numerical part of original data
    - cat_attrs: categorical attributes of the original data
      NOTE: 1D array (could be multi-dimensional?!)
    - combine_attrs: attributes to be combined via the num_pipeline

    Output:
    - a full data pipeline: num_data transformed by num_pipeline,
      cat_attribs transformed by a one-hot encoder.
    """
    num_array = list(df.select_dtypes(exclude="object"))
    cat_array = list(df.select_dtypes(include="object"))

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline(combine_attrs), num_array),
            ("cat", OneHotEncoder(), cat_array),
        ]
    )
    return full_pipeline


def full_transform(df, bins, combine_attrs, labels_to_predict, ratio=0.2):
    """
    Input:
    - df: dataframe
    - ratio: test_set/full_data
    - bins: to split data into train and test set
    - combine_attrs: attributes which will be combined
    - labels_to_predict: used to fit a model.

    Output:
    - dict containing
      + "train set": fully transformed train set using the above functions and
        removing the labels
      + "train labels": the labels (of the train set) which will be used to train
        our models
      + "test set": transformed test set (also removing the labels)
      + "test labels": corresponding labels of test set.
    """
    shuffle_split = shuffle_split_data(df, bins, ratio)
    train_set = shuffle_split[0]
    test_set = shuffle_split[1]

    # get labels and drop them in train and test set
    train_set_drop = train_set.drop(labels_to_predict, axis=1)
    train_labels = train_set[labels_to_predict].copy()
    test_set_drop = test_set.drop(labels_to_predict, axis=1)
    test_labels = test_set[labels_to_predict].copy()

    # transform train_set and test_set (for evaluation)
    full_pipe = full_pipeline(train_set_drop, combine_attrs)
    tr_train_set = full_pipe.fit_transform(train_set_drop)
    tr_test_set = full_pipe.transform(test_set_drop)

    return {
        "train set": tr_train_set,
        "train labels": train_labels,
        "test set": tr_test_set,
        "test labels": test_labels,
    }
