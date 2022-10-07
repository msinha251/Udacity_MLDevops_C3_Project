
# Model Card

## Model Details

Mahesh Sinha created this Model. It is a Random Forest Classifier model trained on census data.


## Model Description

This model predicts whether a person earns over 50k or not based on the census data.


## Model Performance

The model has an ROC AUC score of 0.89 on the validation set with 5-fold cross validation.


## Training Data

The data was collected from the 1994 Census database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics).
Sample data: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data">adult.data</a>
Data Dictionary: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names">adult.names</a>

## Sample Prediction:
```
sample_dict = {     'workclass': 'state_gov',
                    'education': 'bachelors',
                    'marital_status': 'never_married',
                    'occupation': 'adm_clerical',
                    'relationship': 'not_in_family',
                    'race': 'white',
                    'sex': 'male',
                    'native_country': 'united_states',
                    'age': 39,
                    'fnlwgt': 77516,
                    'education_num': 13,
                    'capital_gain': 2174,
                    'capital_loss': 0,
                    'hours_per_week': 40
                }

predict_single(sample_dict, dv, model)
```
        