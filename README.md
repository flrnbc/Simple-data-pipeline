# Simple data pipeline (using sklearn)
The first step to make predictions from data is to sufficiently clean and possibly enhance it (filling in missing values, splitting into train and test data, combine attributes etc.) before applying machine learning techniques. Since this process is similar for various types of data, it is convenient to automate this process via a data pipeline.
This basic data pipeline gives one way to do so using [https://github.com/scikit-learn/scikit-learn][Scikit-learn]]. It has the following functionality:

- loading the data (from the web or locally)
- splitting the data with respect to bins (see below)
- clean the data and add combined attributes.

We next explain these in more detail.


## Loading the data
Via the function ~fetch_data(data_url, data_path, data_name)~ we can download a tgz-file containing a csv-file.
More precisely, we download the file from the URL ~data_url~ to the (relative) path ~data_path/data_name.tgz~ and extract it to ~data_path/data_name.csv~.
Then the function ~load_data_pd(data_path, data_name)~ loads the data to a (Pandas) DataFrame where ~data_path~ and ~data_name~ are as before.


## Splitting the data
Given a DataFrame ~df~ (e.g. ~df = load_data_pd(data_path, data_name)~), we split it into a train and test set. It often makes sense to do so with stratified sampling using so-called bins (see example below). Then ~shuffle_split_data(df, bins, ratio)~ returns a tuple ~(train data, test data)~ such that the ~test data~ has the size ~ratio*(size of df)~.

## Clean and enhance the data
The function ~full_pipeline_tr(df, combine_attrs)~ cleans and enhances the dataframe ~df~ as follows:

- Cleaning the data in two steps. First, ~df~ is split into numerical and categorical attributes ~df_num~ and ~df_cat~ respectively. Secondly, ~df_num~ is cleaned:
 + all 'na' values in a column of ~df_num~ is replaced by the median value of the corresponding column,
 + then ~df_num~ is standarized by removing the mean and scaling to unit variance,
 + ~df_cat~ is one-hot-encoded to obtain a purely numerical array as output.

- Enhancing the data by combining numerical attributes together: ~combine_attrs~ is a list containing integer tuples. For each such tuple (m, n) the function divides the m-th by the n-th numerical attribute (column) and adds result to ~df~ as a new attribute.
