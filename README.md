# Disaster Response Pipeline Project

This project is part of Udacity's Data Scientist Nanodegree program. It aims to analyze disaster data to build a model for an API that classifies disaster messages. The project involves creating an ETL (Extract, Transform, Load) pipeline to clean the data, an ML (Machine Learning) pipeline to train a model, and a web app to predict new messages.

## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Instructions](#Instructions)

## Description
This project consists of several Python scripts:

- **`data/process_data.py`**: This script loads data from an SQLite database, performs ETL on the data, and stores the cleaned data back into the database.
- **`models/train_classifier.py`**: This script loads data from the SQLite database, tokenizes and preprocesses the text data, builds a machine learning model, trains the model using GridSearchCV, evaluates the model, and saves the model as a pickle file.
- **`app/run.py`**: This script runs a web app where users can input messages related to disasters, and the app will classify them into relevant categories using the trained model.

## Installation
To run this project, you'll need Python 3.x and the following libraries installed:

- pandas
- numpy
- scikit-learn
- sqlalchemy
- nltk
- flask

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn sqlalchemy nltk flask
```

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Then, open a web browser and go to http://localhost:3000/ to access the web app.


