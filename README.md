# Simple data pipeline (using sklearn)
The first step to make predictions from data is to sufficiently clean and possibly enhance it (filling in missing values, splitting into train and test data, combine attributes etc.) before applying machine learning techniques. Since this process is similar for various types of data, it is convenient to automate it via a data pipeline.

This basic data pipeline gives one way to do so using [Scikit-learn](https://github.com/scikit-learn/scikit-learn). It has the following functionality:

- loading the data (from the web or locally)
- splitting the data with respect to categories/bins (see below)
- clean the data and add combined attributes.

To ensure this functionality and for future changes, we have added tests (using `pytest`).  

We next explain our data pipeline in more detail.

## Loading data
Via the function

 `fetch_data(data_url, data_dir, data_name)` 

we can download a tgz-file containing a csv-file. More precisely, we download the file from the URL `data_url` to the (relative) path `data_dir/data_name.tgz` and extract it to `data_dir/data_name.csv`. Then the function 

`load_data_pd(data_dir, data_name)` 

loads the data to a (Pandas) DataFrame where `data_dir` and `data_name` are usually as before.

## Splitting data
Given a DataFrame `df` (e.g. `df = load_data_pd(data_dir, data_name)`), we split it into a train and test set. It often makes sense to do so with stratified sampling using so-called categories or bins (see example below). Then

 `shuffle_split_data(df, bins, ratio)` 

returns a tuple `(train_data, test_data)` such that the `test_data` has the size `ratio * (size of df)`.

## Clean and enhance data
After splitting the data as above, we usually continue working with the train data and set `df = train_data`. The function 

`full_pipeline_tr(df, combine_attrs)` 

cleans and enhances the DataFrame `df` as follows:

* First, `df` is split into numerical and categorical attributes `df_num` and `df_cat` respectively. Then `df_num` is cleaned using:
  * all 'na' values in a column of `df_num` is replaced by the median value of the corresponding column,
  * then `df_num` is standarized by removing the mean and scaling to unit variance,
  * `df_cat` is one-hot-encoded to obtain a purely numerical array as output.

* Enhancing the data by combining numerical attributes together: `combine_attrs` is a list containing integer tuples. For each such tuple (m, n) the function divides the m-th by the n-th numerical attribute (column) and adds the result to `df` as a new attribute.

## Example

To showcase our data pipeline, we use the housing dataset, as for example employed in Géron's book "Hands-On Machine Learning..." in Chapter 2 ([repo](https://github.com/ageron/handson-ml2)). 

```python
import SimpleDataPipeline as sdp

# load the data
DATA_NAME = "housing"
DATA_DIR = "datasets/housing/"
DATA_URL = "https://raw.githack.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
housing = sdp.load_data_pd(DATA_DIR, DATA_NAME)

# split data into train and test set
# income categories (bins) for the stratified split
 bins = pd.cut(
        test_data["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5])
shuffle_split = sdp.shuffle_split_data(housing, bins, ratio=0.2)
# only work with train set from now on
housing = shuffle_split[0]

# rooms per household (attributes 3 and 6) and population per household (attributes 5 and 6) 
# as combined attributes
combine_attrs = [(3, 6), (5, 6)]
# transform data with the full pipeline
housing_tr = sdp.full_pipeline_tr(housing, combine_attrs)
```

Then `housing_tr` is a well-prepared dataset (now a numpy array) to which we may apply our ML techniques. 

(Note: The url given in Géron's book (2nd edition) did not work for me. But it did with raw.githack.)

 