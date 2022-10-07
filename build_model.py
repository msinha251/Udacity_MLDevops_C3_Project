from operator import mod
import pandas as pd
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import pickle

#logging:
logging.basicConfig(level=logging.INFO)


def load_data(data_path):
    '''
    Load data from data_path
    '''
    #extracting columns from adult.names
    cols = []
    logging.info(f'Extracting columns from adult.names')
    with open(f'{data_path}/adult.names', 'r') as names:
        for line in names:
            if ':' in line and '|' not in line:
                cols.append(line.split(':')[0])
    logging.info(f'Columns: {cols}')

    #load train data:
    df_train = pd.read_csv(f'{data_path}/adult.data', names=cols+['earn_over_50k'], index_col=False)
    logging.info(f'Train data shape: {df_train.shape}')
    #load test data:
    df_test = pd.read_csv(f'{data_path}/adult.test', names=cols+['earn_over_50k'], index_col=False, skiprows=[0])
    logging.info(f'Test data shape: {df_test.shape}')

    #convert train target to 0/1:
    df_train['earn_over_50k'] = df_train['earn_over_50k'].apply(lambda x: 1 if x == ' >50K' else 0)
    logging.info(f'Train target counts: {df_train["earn_over_50k"].value_counts()}')
    #convert test target to 0/1:
    df_test['earn_over_50k'] = df_test['earn_over_50k'].apply(lambda x: 1 if x == ' >50K.' else 0)
    logging.info(f'Test target counts: {df_test["earn_over_50k"].value_counts()}')

    return df_train, df_test

def basic_cleaning(df):
    '''
    Basic cleaning of data-
    '''
    #fixing column names:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    logging.info(f'Columns: {df.columns}')

    #filter categorical columns and numerical columns:
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    logging.info(f'Categorical columns: {cat_cols}')
    logging.info(f'Numerical columns: {num_cols}')
    
     #replacing spaces & - with underscore in categorical columns:
    for col in cat_cols:
        df[col] = df[col].str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    #replacing ? with nan:
    logging.info(f'Before replacing ? with nan: {df.isin(["?"]).sum()}')
    df = df.replace('?', np.nan)
    logging.info(f'Nan values: {df.isna().sum()}')

    #fill nan with mode for categorical columns:
    logging.info(f'Filling nan with mode for categorical columns')
    for col in cat_cols:
        if col != 'earn_over_50k':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            cat_cols.remove(col)

    #fill nan with mean for numerical columns:
    for col in num_cols:
        if col != 'earn_over_50k':
            df[col] = df[col].fillna(df[col].mean())
        else:
            num_cols.remove(col)

    return df, cat_cols + num_cols


def train_model(df_train, cols):
    '''
    Trains model on train data
    '''
    logging.info(f'Training model on Columns: {cols}')
    
    #Initialize KFold:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    fold = 0
    for train_index, val_index in kf.split(df_train):
        logging.info(f'Fold: {fold} -- Train samples: {len(train_index)} -- Validation samples: {len(val_index)}')

        #create train dict:
        train_dict = df_train[cols].iloc[train_index].to_dict(orient='records')
        
        #create val dict:
        val_dict = df_train[cols].iloc[val_index].to_dict(orient='records')
        
        #create train target:
        train_target = df_train['earn_over_50k'].iloc[train_index]

        #create val target:
        val_target = df_train['earn_over_50k'].iloc[val_index]

        #initialize DictVectorizer:
        dv = DictVectorizer(sparse=False)

        #fit DictVectorizer on train dict:
        dv.fit(train_dict)

        #transform train dict:
        X_train = dv.transform(train_dict)

        #transform val dict:
        X_val = dv.transform(val_dict)

        #initialize & fit LogisticRegression:
        #model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42) 
        model.fit(X_train, train_target)

        #predict on val data:
        y_pred = model.predict_proba(X_val)[:, 1]

        #calculate roc_auc_score:
        score = roc_auc_score(val_target, y_pred)
        scores.append(score)

        logging.info(f'Fold: {fold}, Score: {score}')
        fold += 1

    logging.info(f'Mean Score: {np.mean(scores)}')

    return dv, model

def save_model(dv, model, model_path):
    '''
    Save model to model_path
    '''
    #save DictVectorizer:
    with open(f'{model_path}/dv.bin', 'wb') as f_out:
        pickle.dump(dv, f_out)
        f_out.close()
    logging.info(f'DictVectorizer saved')

    #save model:
    with open(f'{model_path}/model1.bin', 'wb') as f_out:
        pickle.dump(model, f_out)
        f_out.close()
    logging.info(f'Model saved')

def save_model_card(model_path):
    '''
    Save model card to model_path
    '''
    #Write model card:
    with open(f'{model_path}/model_card.md', 'w') as f_out:
        f_out.write('''
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
        ''')
        f_out.close()
    logging.info(f'Model card saved to {model_path}')



def predict_batch(df_test, dv, model, cols):
    '''
    Predicts on test data
    '''
    #create test dict:
    test_dict = df_test[cols].to_dict(orient='records')

    #transform test dict:
    X_test = dv.transform(test_dict)

    #predict on test data:
    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred

def predict_single(sample_dict, dv, model):
    '''
    Predicts on single row of test data
    Sample input:
    {   'workclass': 'state_gov',
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
    '''

    #transform test dict:
    X_test = dv.transform(sample_dict)

    #predict on test data:
    y_pred = model.predict_proba(X_test)[:, 1]

    return y_pred[0]

def load_model(model_path):
    '''
    Load model from model_path
    '''
    #load DictVectorizer:
    with open(f'{model_path}/dv.bin', 'rb') as f_in:
        dv = pickle.load(f_in)
        f_in.close()
    logging.info(f'DictVectorizer loaded')

    #load model:
    with open(f'{model_path}/model1.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    logging.info(f'Model loaded')
    return dv, model


'''
Write a function that computes performance on model slices. I.e. a function that computes the performance metrics when the value of a given feature is held fixed. E.g. for education, it would print out the model metrics for each slice of data that has a particular value for education. You should have one set of outputs for every single unique value in education.
Complete the stubbed function or write a new one that for a given categorical variable computes the metrics when its value is held fixed.
Write a script that runs this function (or include it as part of the training script) that iterates through the distinct values in one of the features and prints out the model metrics for each value.
Output the printout to a file named slice_output.txt.

'''
def slice_metrics(df, dv, model, cols, feature):
    '''
    Computes performance on model slices.
    '''
    #get unique values of feature:
    feature_values = df[feature].unique()

    with open('slice_output.txt', 'w') as f_out:
        for value in feature_values:
            #filter df:
            df_slice = df[df[feature] == value]

            #create test dict:
            test_dict = df_slice[cols].to_dict(orient='records')

            #transform test dict:
            X_test = dv.transform(test_dict)

            #predict on test data:
            y_pred = model.predict_proba(X_test)[:, 1]
            
            #calculate roc_auc_score:
            score = roc_auc_score(df_slice.earn_over_50k, y_pred)

            f_out.write(f'{feature} = {value}: {score}')
            logging.info(f'{feature} = {value}: {score}')
        f_out.close() 






if __name__ == '__main__':
    #load data:
    df_train, df_test = load_data('./data')
    #clean data:
    df_train, cols = basic_cleaning(df_train)
    df_test, _ = basic_cleaning(df_test)
    logging.info(f'Train data shape: {df_train.shape}')

    #train model:
    dv, model = train_model(df_train, cols)

    #predict on test data:
    y_pred = predict_batch(df_test, dv, model, cols)
    logging.info(f'Score on test data: {roc_auc_score(df_test.earn_over_50k, y_pred)}')
    
    #compute_metrics on slice feature:
    slice_metrics(df_test, dv, model, cols, 'education')

    #save predictions:
    df_test['earn_over_50k'] = y_pred
    df_test[['earn_over_50k']].to_csv('./data/predictions.csv', index=False)
    logging.info(f'Predictions saved')

    #save model & model card:
    save_model(dv, model, './model')
    save_model_card('./model')

    logging.info('Done')
        




