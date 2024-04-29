# MLSMOTE
Multi label Synthetic Minority Over-sampling Technique (MLSMOTE)

**Changes:**

* Updated to work correctly with pandas DataFrames as input by converting to numpy arrays before resampling and back to DataFrames after resampling
* Added `input_columns` and `label_columns` arguments to specify the input columns and labels for resampling

For details, go through https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87 and https://github.com/scikit-learn-contrib/imbalanced-learn/blob/70e1f966414cdf9425d764c52a0a92d99c732051/imblearn/over_sampling/_mlsmote.py
