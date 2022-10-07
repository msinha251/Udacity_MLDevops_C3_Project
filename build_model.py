import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def load_data(data_path):
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
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    #fill nan with mean for numerical columns:
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    return df

if __name__ == '__main__':
    df_train, df_test = load_data('./data')
    df_train = basic_cleaning(df_train)
    df_test = basic_cleaning(df_test)
    logging.info(f'Train data shape: {df_train.shape}')