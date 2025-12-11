# Health Insurance Premium Prediction

The Health Insurance Premium Predition system gives an estimate of the Health Insurance Premiums for the user profile. It takes user data as input and uses random forest regression to predict premium price estimates as output. The Health Insurance sub-package has three modules namely -
- `preprocessing.py`
- `training.py`
- `result.py`
# Input and Output
### Input -
- Age
- Sex
- BMI
- Number of Dependents
- Smoking
- Region
### Output - 
- Health Insurance Premium Estimates
# Module: `preprocessing.py`
This module preprocesses the dataset (insurance_data.csv) and makes it ready for training. This module contains the class `preprocess` which contains methods like `train_test()` and `preprocessing()`. The `train_test()` function as the name suggests convert the csv data into a pandas dataframe and splits it into training and test sets. On the other hand `preprocessing()` converts the categorical predictors like sex, smoker, and region into OneHotEncoded Columns.
# Module: `training.py`
This module is used to train the data that was previous preprocessed by `preprocessing.py`. This Module contains two functions `Training()` and `save()`. The `Training()` function calls the `preprocess` class from the previous module and uses that data produced to fir a Random Forest Regression Tree. This model is then evaluated to find $RMSE$ and $R^2$ values using test set. The `save()` function then uses the joblib package to save the random forest model as a .pkl file.
# Module: `result.py`
The final module in this subpackage is the `result.py` module that has the class `result` that inherits the class `preprocess` from the first module. The `result` class takes in the input data and uses the method `predict()` to output the Health Insurance Premium Estimate.
# Requirements
The requirements for this subpackage involves -
- pandas
- joblib
- sci-kit learn or sklearn
# Demo Output
![Health Insurance](Output/Health_Insurance.png)
