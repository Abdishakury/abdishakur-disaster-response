# import libraries
import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath(str) - the path of the messages .csv file
    categories_filepath(str) - the path of the categories .csv file

    OUTPUT:
    df(dataframe) - dataframe of messages table and categories table merged on id
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype=str, encoding='utf-8')
    # load categories dataset
    categories = pd.read_csv(categories_filepath, dtype=str, encoding='ascii')
    # merge datasets
    df = pd.merge(messages,categories, how='inner', on='id')
    return df

def clean_data(df):
    '''
    INPUT:
    df(dataframe) - dataframe of messages table and categories table merged on id

    OUTPUT:
    df(dataframe) - dataframe cleaned
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    # categories.head()

    # Convert category values to just numbers 0 or 1
    categories.related.apply(lambda x: x[-1])
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # categories.head()

    # check if there are values other than 0 and 1 in categories
    others = []
    for col in categories.columns:
        others.append(categories[col].unique())

    # If there are values other than 0 or 1 in 'categories' columns, replace 
    # other values as 1
    for col in categories.columns:
        categories.loc[(categories[col]!=1)&(categories[col]!=0)] = 1

    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # check number of duplicates
    # df.duplicated().sum()

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # check number of duplicates
    # df.duplicated().sum()

    return df

def save_data(df, database_filename):
    '''Function to save the clean dataset into an sqlite database
    INPUT:
    df(dataframe) - the cleaned dataframe
    database_filename(str) - the name of the stored .db file
    '''
    # create SQL engine with database name
    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    # extract table name from database name
    table_name = os.path.basename(database_filename).split('.')[0]

    # save the clean dataset into an sqlite database
    df.to_sql(table_name, engine, index=False, if_exists='replace') 

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()