# MLSMOTE
Multi label Synthetic Minority Over-sampling Technique (MLSMOTE)

**Changes:**

* Updated to work correctly with pandas DataFrames as input by converting to numpy arrays before resampling and back to DataFrames after resampling
* Added `input_columns` and `label_columns` arguments to specify the input columns and labels for resampling

For details, go through https://medium.com/thecyphy/handling-data-imbalance-in-multi-label-classification-mlsmote-531155416b87
