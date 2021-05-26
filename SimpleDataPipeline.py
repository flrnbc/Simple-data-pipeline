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

# TODO: add labels and drop


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


def full_pipeline_tr(df, combine_attrs):
    """Transforms the dataframe df via full_pipeline."""
    # split dataframe df into numerical (num_array) and categorical (cat_array)

    # get the pipeline
    full_pipe = full_pipeline(df, combine_attrs)

    return full_pipe.fit_transform(df)
