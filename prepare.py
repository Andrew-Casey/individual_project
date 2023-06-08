import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt

def get_prepped_pga():
    """
    Reads and prepares the PGA data set for analysis.

    Returns:
        pd.DataFrame: A preprocessed DataFrame containing the PGA data.

    The function performs the following steps:
    1. Reads the 'PGA_data.csv' file into a DataFrame.
    2. Drops rows with missing values (nulls).
    3. Converts all column names to lowercase.
    4. Drops a specific set of columns that are not relevant to the study.
    5. Renames selected columns for better readability.
    6. Converts the 'date' column to datetime format.

    Note:
        - The 'PGA_data.csv' file must be present in the current directory.
        - The returned DataFrame is ready for further analysis.

    Example:
        df = get_prepped_pga()
        # Perform analysis on the preprocessed data
    """
    # get data set
    df = pd.read_csv('PGA_data.csv', index_col=0) 

    # drop all nulls
    df = df.dropna()

    #lower case all column names
    df.columns  = df.columns.str.lower()

    # this study focuses on performance the previous week. Therefore, I am dropping the following columns
    df = df.drop(columns=['height cm', 'weight lbs', 'dob', 'age','player id', 'tournament id',
                'season', 'visibility', 'winddirdegree', 'windspeedkmph', 'greensgrass','fariwaysgrass', 
                'water','bunkers', 'windchillc','windgustkmph', 'cloudcover', 'humidity','precipmm','pressure',
                'tempc', 'final position', 'major', 'consecutive_cuts_made', 'finish','moonrise', 'sunrise',
                'sunset', 'dewpointc', 'feelslikec', 'heatindexc', 'maxtempc','mintempc','totalsnow_cm', 'sunhour',
                'uvindex', 'moon_illumination', 'moonset', 'place','number of rounds', 'sg_putt','sg_arg','sg_app',
                'sg_ott', 'sg_t2g', 'sg_total', 'score', 'slope','length','par'])

    # rename columns
    df = df.rename(columns={'tournament name':'tournament_name', 'drive yards':'driving_avg',
                        'fairways hit':'fairways_hit', 'putts/hole':'putting_avg'})

    #change tournament date to datetime
    df.date = pd.to_datetime(df.date)   

    return df

def split_data(df, target):
    '''
    take in a DataFrame and target variable. return train, validate, and test DataFrames; stratify on target variable.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    # reset index
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, validate, test

def scaled_df(train, validate, test):
    """
    This function scales the train, validate, and test data using the MinMaxScaler.

    Parameters:
    train (pandas DataFrame): The training data.
    validate (pandas DataFrame): The validation data.
    test (pandas DataFrame): The test data.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_validate_scaled (pandas DataFrame): The scaled validation data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        y_train (pandas Series): The target variable for the training data.
        y_validate (pandas Series): The target variable for the validation data.
        y_test (pandas Series): The target variable for the test data.
    """

    X_train = train[['sg_putt_prev','sg_arg_prev','sg_app_prev','sg_ott_prev','sg_t2g_prev','sg_total_prev','driving_avg','fairways_hit','putting_avg']]
    X_validate = validate[['sg_putt_prev','sg_arg_prev','sg_app_prev','sg_ott_prev','sg_t2g_prev','sg_total_prev','driving_avg','fairways_hit','putting_avg']]
    X_test = test[['sg_putt_prev','sg_arg_prev','sg_app_prev','sg_ott_prev','sg_t2g_prev','sg_total_prev','driving_avg','fairways_hit','putting_avg']]

    y_train = train.made_cut
    y_validate = validate.made_cut
    y_test = test.made_cut

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler 
    # AND!!!!
    #using the scaler on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # Convert the array to a DataFrame
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_validate.columns, index=X_validate.index)
    
    # Convert the array to a DataFrame
    X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    #distrinution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head()}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
# fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    num_cols = len(get_numeric_cols(df))
    num_rows, num_cols_subplot = divmod(num_cols, 3)
    if num_cols_subplot > 0:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

    for i, col in enumerate(get_numeric_cols(df)):
        row_idx, col_idx = divmod(i, 3)
        sns.histplot(df[col], ax=axes[row_idx, col_idx])
        axes[row_idx, col_idx].set_title(f'Histogram of {col}')
                                         
    plt.tight_layout()
    plt.show() 

def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing

def nulls_by_row(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({
                    'num_cols_missing': num_missing,
                    'percent_cols_missing': pct_miss
                    })
    
    return rows_missing

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

def scaled_df_2(train, validate, test):
    """
    This function scales the train, validate, and test data using the MinMaxScaler.

    Parameters:
    train (pandas DataFrame): The training data.
    validate (pandas DataFrame): The validation data.
    test (pandas DataFrame): The test data.

    Returns:
    Tuple of:
        X_train_scaled (pandas DataFrame): The scaled training data.
        X_validate_scaled (pandas DataFrame): The scaled validation data.
        X_test_scaled (pandas DataFrame): The scaled test data.
        y_train (pandas Series): The target variable for the training data.
        y_validate (pandas Series): The target variable for the validation data.
        y_test (pandas Series): The target variable for the test data.
    """

    X_train = train[['sg_putt_2wk_avg','sg_arg_2wk_avg','sg_app_2wk_avg','sg_ott_2wk_avg','sg_t2g_2wk_avg','sg_total_2wk_avg','driving_avg','fairways_hit','putting_avg']]
    X_validate = validate[['sg_putt_2wk_avg','sg_arg_2wk_avg','sg_app_2wk_avg','sg_ott_2wk_avg','sg_t2g_2wk_avg','sg_total_2wk_avg','driving_avg','fairways_hit','putting_avg']]
    X_test = test[['sg_putt_2wk_avg','sg_arg_2wk_avg','sg_app_2wk_avg','sg_ott_2wk_avg','sg_t2g_2wk_avg','sg_total_2wk_avg','driving_avg','fairways_hit','putting_avg']]

    y_train = train.made_cut
    y_validate = validate.made_cut
    y_test = test.made_cut

    #making our scaler
    scaler = MinMaxScaler()
    #fitting our scaler 
    # AND!!!!
    #using the scaler on train
    X_train_scaled = scaler.fit_transform(X_train)
    #using our scaler on validate
    X_validate_scaled = scaler.transform(X_validate)
    #using our scaler on test
    X_test_scaled = scaler.transform(X_test)

    # Convert the array to a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # Convert the array to a DataFrame
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=X_validate.columns, index=X_validate.index)
    
    # Convert the array to a DataFrame
    X_test_scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

def get_prepped_pga2():
    """
    Reads and prepares the PGA data set for analysis.

    Returns:
        pd.DataFrame: A preprocessed DataFrame containing the PGA data.

    The function performs the following steps:
    1. Reads the 'PGA_data.csv' file into a DataFrame.
    2. Drops rows with missing values (nulls).
    3. Converts all column names to lowercase.
    4. Drops a specific set of columns that are not relevant to the study.
    5. Renames selected columns for better readability.
    6. Converts the 'date' column to datetime format.

    Note:
        - The 'PGA_data.csv' file must be present in the current directory.
        - The returned DataFrame is ready for further analysis.

    Example:
        df = get_prepped_pga()
        # Perform analysis on the preprocessed data
    """
    # get data set
    df = pd.read_csv('PGA_data.csv', index_col=0) 

    # drop all nulls
    df = df.dropna()

    #lower case all column names
    df.columns  = df.columns.str.lower()

    # this study focuses on performance over two weeks. Therefore, I am dropping the following columns
    df = df.drop(columns=['height cm', 'weight lbs', 'dob', 'age','player id', 'tournament id',
                'season', 'visibility', 'winddirdegree', 'windspeedkmph', 'greensgrass','fariwaysgrass', 
                'water','bunkers', 'windchillc','windgustkmph', 'cloudcover', 'humidity','precipmm','pressure',
                'tempc', 'final position', 'major', 'consecutive_cuts_made', 'finish','moonrise', 'sunrise',
                'sunset', 'dewpointc', 'feelslikec', 'heatindexc', 'maxtempc','mintempc','totalsnow_cm', 'sunhour',
                'uvindex', 'moon_illumination', 'moonset', 'place','number of rounds', 'score', 'slope','length','par'])

    # rename columns
    df = df.rename(columns={'tournament name':'tournament_name', 'drive yards':'driving_avg',
                        'fairways hit':'fairways_hit', 'putts/hole':'putting_avg'})

    #change tournament date to datetime
    df.date = pd.to_datetime(df.date)   

    # feature engineer 2wk performance average
    df['sg_putt_2wk_avg'] = (df.sg_putt + df.sg_putt_prev)/2
    df['sg_arg_2wk_avg'] = (df.sg_arg + df.sg_arg_prev)/2
    df['sg_app_2wk_avg'] = (df.sg_app + df.sg_app_prev)/2
    df['sg_ott_2wk_avg'] = (df.sg_ott + df.sg_ott_prev)/2
    df['sg_t2g_2wk_avg'] = (df.sg_t2g + df.sg_t2g_prev)/2
    df['sg_total_2wk_avg'] = (df.sg_total + df.sg_total_prev)/2

    #drop previous and current weeks performance columns
    df = df.drop(columns = ['sg_putt','sg_arg','sg_app','sg_ott','sg_t2g','sg_total',   
                        'sg_putt_prev','sg_arg_prev','sg_app_prev','sg_ott_prev',
                        'sg_t2g_prev','sg_total_prev'])


    return df