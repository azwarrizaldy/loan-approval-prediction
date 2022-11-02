## Final Project Data Expert G2Academy | author : azwar8597@gmail.com

## Loan Approval Prediction
This project describes how to build a machine learning model for loan approval prediction where data is retrieved from AWS RDS PostgreSQL and stored in AWS S3.

## The process steps are as follows:

### 1. Read Data from AWS RDS PostgreSQL
  - connect to AWS RDS PostgreSQL
  - read and get data from AWS RDS PostgreSQL
  - drop unnecessary column 

### 2. Explanatory Data Analysis (EDA)
  - read data
  - process descriptive statistic analysis
  - create plot

### 3. Training Machine Learning Model
  - read data
  - check missing value
  - imputation of missing values
  - encoding for variable categorical
  - selection feature
  - define dependent and independet variable
  - split data
  - classification with algorithm KNN
  - print classification report
  - save model to local directory
  - upload model to AWS S3 Bucket

## Requirements
  - pandas
  - psycopg2
  - sqlalchemy
  - python-dotenv
  - matplotlib
  - numpy
  - pickle
  - datetime
  - boto3
  - scikit-learn

## Source Data
from Kaggle :
https://www.kaggle.com/datasets/sethirishabh/finance-company-loan-data
