"""
    author  : azwar8597@gmail.com
    project : Machine Learning KNN 
"""

import pandas as pd #type: ignore
import numpy as np #type: ignore
import pickle5 as pickle #type: ignore
from datetime import datetime as dt #type: ignore
from sklearn.neighbors import KNeighborsClassifier #type: ignore
from sklearn.metrics import accuracy_score, classification_report #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from sklearn import metrics #type: ignore
from dotenv import load_dotenv #type: ignore
import string, re, os, os.path #type: ignore
import boto3 #type: ignore
from . read import ReadData



class TrainingData:


    def train(self):

        #========================== Data Preprocessing ==========================

        #read data
        df_data = ReadData.read_data(self)

        #manipulation data for categorical variables
        df_data['gender'].fillna(df_data['gender'].mode()[0],inplace=True)
        df_data['married'].fillna(df_data['married'].mode()[0],inplace=True)
        df_data['dependents'].fillna(df_data['dependents'].mode()[0],inplace=True)
        df_data['self_employed'].fillna(df_data['self_employed'].mode()[0],inplace=True)
        df_data['credit_history'].fillna(df_data['credit_history'].mode()[0],inplace=True)
        df_data['loan_amount_term'].fillna(df_data['loan_amount_term'].mode()[0],inplace=True)

        #manipulation data for numerical variables
        df_data['loan_amount'].fillna(df_data['loan_amount'].mean(),inplace=True)

        #encoder label or convert categorical to numerical
        for x in df_data.columns:
            if df_data[x].dtype == np.number:
                continue
  
            #implementation encoding for each variable categorical
            df_data[x] = LabelEncoder().fit_transform(df_data[x])
        
        #selection feature
        df_data = df_data.drop(["customer_id"], axis=1)

        #========================== Fitting Model ==========================

        #define independent dan dependent features
        X = df_data.drop(['loan_status'], axis=1)
        y = df_data['loan_status']

        #split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #determine value of K
        score_list = []
        error_rate = []

        for k in range(1, 100):
                #define algorithm knn
                knn = KNeighborsClassifier(n_neighbors=k)
                #fitting model
                knn = knn.fit(X_train, y_train)
                #predict data
                pred_k = knn.predict(X_test)
                
                #append accuracy and error rete
                score_list.append(metrics.accuracy_score(y_test, pred_k))
                error_rate.append(np.mean(pred_k != y_test))

        k_score = score_list.index(max(score_list))
        k_error = error_rate.index(min(error_rate))

        #Determine Value K
        if k_score == k_error:
            k_neighbors = k_error
        else:
            k_neighbors = k_score
        
        #selection and fitting model
        model_knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        model_knn = model_knn.fit(X, y)

        #prediction test model
        pred = model_knn.predict(X_test)

        #print accuracy score
        print("Report : \n", classification_report(y_test, pred))

        #format name as datetime
        model_name = dt.now().strftime("%Y%m%d%H%M%S%f")

        #create absolute path for save model
        model_clf = "./application/model/{}.clf".format(model_name)

        #save model to local disk
        if model_clf is not None:
            with open(model_clf, 'wb') as f:
                pickle.dump(model_knn, f)
    
        #path model in directory 
        ModelDir = "./application/model/{}.clf"

        try:
            
            #for check model in folder directory
            models = [
                        dt.strptime(model.replace('.clf', ''), "%Y%m%d%H%M%S%f")
                        for model in os.listdir("./application/model/")
                        if model.endswith('.clf')
                ]
            
        except:

            print("error")
            
        else:
            
            if not models:
                print("model not found")

            else:

                #sort file name in directory
                models.sort()

                #path directory latest model
                model_path = ModelDir.format(
                models[-1].strftime("%Y%m%d%H%M%S%f")
                )

                #format model name
                file_name = '{}.clf'.format(models[-1].strftime("%Y%m%d%H%M%S%f"))
                
                #load function dotenv
                load_dotenv()

                #define env aws access key id
                KEY_ID          = os.environ['key']

                #define env aws secret access key
                ACCESS_KEY      = os.environ['acc']

                #define env region name
                REGION          = os.environ['reg']

                #define env S3 bucket name
                BUCKET          = os.environ['buc']

                #initiation connect to S3
                s3 = boto3.client(
                            's3',
                            aws_access_key_id = '{}'.format(KEY_ID),
                            aws_secret_access_key = '{}'.format(ACCESS_KEY),
                            region_name = '{}'.format(REGION)
                        )
                    
                #upload model to S3
                s3.upload_file(model_path, '{}'.format(BUCKET), 'model/{}'.format(file_name))