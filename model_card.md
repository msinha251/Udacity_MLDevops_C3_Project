# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Mahesh Sinha created this Model. It is a Random Forest Classifier model trained on census data.

## Intended Use
This model predicts whether a person earns over 50k or not based on the census data.

## Training Data
The data was collected from the 1994 Census database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). 
Training data is used from <a href='https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv'> here </a> <br>
Sample observation: 
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
Evaluation data is the 20% of full downloaded data by using `train_test_split` function from scikit-learn.
Model has been evaluated by *fbeta*, *precision* and *recall* scores. And the score on Evaluation data is below:
`Precision: 0.7250187828700225, Recall: 0.619781631342325, Fbeta: 0.6682825484764543`

## Ethical Considerations
This model is trained on census data. The model is not biased towards any particular group of people.

## Caveats and Recommendations
This model is not suitable for real-time predictions. It is suitable for batch predictions.