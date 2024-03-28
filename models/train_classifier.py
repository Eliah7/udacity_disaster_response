import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Load data from an SQLite database into a pandas DataFrame.

    Args:
        database_filepath (str): Filepath of the SQLite database.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded data.
    """
    engine = create_engine(f'sqlite:///{database_filepath}.db')
    query = "SELECT * FROM categories;"
    df = pd.read_sql(query, engine)
    print(df.head())
    X = df.message.values
    Y = df.genre.values    
    category_names = list(df.columns)[3:]
    print(X, Y, category_names)
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize text data.

    Args:
        text (str): Text data to be tokenized.

    Returns:
        list: List of tokens.
    """
    pass


def build_model():
    """
    Build and return a machine learning model.

    Returns:
        Model: Machine learning model.
    """
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a machine learning model.

    Args:
        model: Machine learning model to be evaluated.
        X_test: Test features.
        Y_test: True labels for the test data.
        category_names (list): List of category names.

    Returns:
        None
    """
    pass


def save_model(model, model_filepath):
    """
    Save a machine learning model to a file.

    Args:
        model: Machine learning model to be saved.
        model_filepath (str): Filepath to save the model.

    Returns:
        None
    """
    pass


def main():
    """
    Main function to train and evaluate a machine learning model.

    Reads command line arguments, loads data, trains the model, evaluates it, and saves the       model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()