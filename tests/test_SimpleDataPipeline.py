import numpy as np
import pandas as pd
import SimpleDataPipeline as sdp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data_path = "datasets"
data_url = (
    "https://raw.githack.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
)
data_name = "housing"

sdp.fetch_data(data_url, data_path, data_name)
test_data = sdp.load_data_pd(data_path, data_name)


def test_shuffle_split_data():
    # test_data_short = test_data.iloc[:100]
    bins = pd.cut(
        test_data["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    strat_test_data = sdp.shuffle_split_data(test_data, bins, ratio=0.2)

    assert len(strat_test_data[0]) == 0.8 * len(test_data)
    assert len(strat_test_data[1]) == 0.2 * len(test_data)


def test_CombinedAttributesAdder():
    adder = sdp.CombinedAttributesAdder([(0, 1), (1, 2)])

    X = np.array([[2, 4, 8],
                  [2, 4, 8],
                  [4, 8, 16]])
    X_adder = adder.fit_transform(X)

    Y = np.array([[0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5]])

    assert (X_adder == np.c_[X, Y]).all()


def test_num_pipeline():
    pipeline_num = sdp.num_pipeline([(0, 1), (1, 2)])

    X = np.array([[2, 4, 8, np.nan],
                  [2, 4, 8, np.nan],
                  [4, 8, 16, np.nan]])

    # manually execute the steps in the pipeline
    imputer = SimpleImputer(strategy="median")
    X1 = imputer.fit_transform(X)
    # combine attributes
    Y = np.array([[0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5]])
    X2 = np.c_[X1, Y]
    # scale
    scaler = StandardScaler()
    X3 = scaler.fit_transform(X2)

    assert (X3 == pipeline_num.fit_transform(X)).all()


def test_full_pipelin_tr():
    X_df = pd.DataFrame([[2, 4, 8, 6, "a"],
                         [2, 4, 8, 8, "b"],
                         [4, 8, 16, np.nan, "c"]])
    # numerical part
    X_num = np.array([[2, 4, 8, 6],
                      [2, 4, 8, 8],
                      [4, 8, 16, np.nan]])
    # categorical part already one-hot-encoded
    X_cat_1hot = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
    pipeline_num = sdp.num_pipeline([(0, 1), (1, 2)])
    X1 = pipeline_num.fit_transform(X_num)
    Y = np.c_[X1, X_cat_1hot]

    # now apply full pipeline
    X_tr = sdp.full_pipeline_tr(X_df, [(0, 1), (1, 2)])

    assert (X_tr == Y).all()
