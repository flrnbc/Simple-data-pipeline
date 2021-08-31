# Simple data pipeline (using sklearn)
The first step to learn from data is to sufficiently clean and possibly enhance it (filling in missing values, splitting into train and test data, combine attributes etc.). Since this process is similar for various types of data, it is convenient to automate it via a data pipeline.

This basic data pipeline gives one way to do so using [Scikit-learn](https://github.com/scikit-learn/scikit-learn). It has the following functionality:

- loading the data (from the web or locally)
- splitting the data with respect to categories/bins (see below)
- clean the data and add combined attributes.

To ensure this functionality and for future changes, we have added some tests using `pytest` (more to be added).  

We next explain our data pipeline in more detail.

## Loading data
Via the function

 `fetch_data(data_url, file_dir)` 

we can download (archived) csv-files. More precisely, we download the file from the URL `data_url` to the (relative) directory `file_dir` (and extract it there) if needed. 

## Transforming data 
Assume we have imported the downloaded csv-files as a Pandas DataFrame `df` (with labeled columns). Further we want to train ML models to predict one of the columns named `to_predict`. The function 

`full_transform(df, bins, combine_attrs, to_predict, ratio)` 

prepares the data for this task. Here `bins` (technically a Pandas Series) used to shuffle split the data into train and test set. Moreover, `combine_attrs` is a list if tuples `(n, m)` such that the n- and m-th column will be combined to a new attribute (these columns have to contain numerical values only). This is often useful to meaningfully enhance our data. Finally, `ratio` is the size of test_set by the size of df.

With this input, `full_transform` does the following:

1. Shuffle split the data `df` using `bins` into train and test set.
2. Split off the labels (the column `to_predict`) from train and test set.
3. Transform the train set as follows: 
   * split it into numerical and categorical attributes and one-hot-encoded the latter,
   * clean the numerical data by replacing 'na' values with the median value of the corresponding column and standarize it.  
4. Apply the same transformations to the test set.

Finally, we obtain the transformed data as dictionary with the self-explaining keys 

* "transformed_train_set"
* "train_labels"
* "transformed_test_set"
* "test_labels".

Now we are ready to train ML models!


## Example

To showcase our data pipeline, we use the housing dataset, as for example employed in Géron's book "Hands-On Machine Learning..." in Chapter 2 ([repo](https://github.com/ageron/handson-ml2)). Please see `Example.ipynb` for details.

 
