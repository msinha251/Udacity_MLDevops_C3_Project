import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def basic_cleaning(df, output_path, target, test=False):
    '''
    Basic cleaning of data
    '''
    logging.info("Cleaning data")

    #remove spaces from column names
    df.columns = df.columns.str.replace(" ", "")

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
        if col != target:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            cat_cols.remove(col)

    #fill nan with mean for numerical columns:
    for col in num_cols:
        if col != target:
            df[col] = df[col].fillna(df[col].mean())
        else:
            num_cols.remove(col)
    logging.info(f'After filling nan: {df.isna().sum()}')

    #save cleaned data:
    if test==False:
        try:
            df.to_csv(output_path, index=False)
            logging.info(f'Cleaned data saved to {output_path}')
            return df, cat_cols, num_cols
        except:
            logging.error(f'Unable to save data to {output_path}')
    else:
        logging.info(f'Cleaned data returned')
        return df, cat_cols, num_cols
