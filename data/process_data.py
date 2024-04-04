import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.

    Returns:
        pandas.DataFrame: Merged dataframe containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories = pd.read_csv(categories_filepath, encoding='latin-1')
    
    df = pd.merge(messages, categories, on='id')
    print("DF =>", df.head())
    
    return df
    
    


def clean_data(df):
    """
    Clean the merged DataFrame.

    Args:
        df (pandas.DataFrame): Merged dataframe containing messages and categories.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    categories = df.categories.str.split(";", expand=True)
    print('CATEGORIES => ', categories.head())
    
    row = categories.iloc[0]
    category_colnames = [i[:-2] for i in row]
    print(category_colnames)
    
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = [cat[len(cat)-1:] for cat in categories[column]]
        categories[column] = pd.Series(categories[column], dtype="object")
    categories = categories[categories['storm'] != 2]
    print("CATEGORES DF => ", categories.head())
    
    df.drop(['categories'], inplace=True, axis=1)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    print("Duplicate Count => ", df.duplicated().sum())
    
    df.drop_duplicates(inplace=True)
    
    print("Duplicate Count => ", df.duplicated().sum())
    print("DF COLS => ", df.columns)
    
    
    return df

def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.

    Args:
        df (pandas.DataFrame): Cleaned dataframe.
        database_filename (str): Filepath for the output SQLite database.
        
    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql('categories', engine, if_exists='replace', index=False)


def main():
    """
    Main function to orchestrate the data processing pipeline.

    Reads command line arguments, loads data, cleans it, and saves it to a database.
    """
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