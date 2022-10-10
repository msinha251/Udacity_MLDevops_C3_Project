# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Mahesh Sinha created this Model. It is a Random Forest Classifier model trained on census data.

## Intended Use
This model predicts whether a person earns over 50k or not based on the census data.

## Training Data
The data was collected from the 1994 Census database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). 
Sample data: 
```
{   
    'workclass': 'state_gov',
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
```

## Evaluation Data
Sample Evaluation data: 
```
{   
    'workclass': 'state_gov',
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
```
