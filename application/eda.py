"""
    author  : azwar8597@gmail.com
    project : Machine Learning KNN 
"""


import pandas as pd #type: ignore
import numpy as np #type: ignore
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore
from . read import ReadData

class EdaData:


    def eda_data(self):

        #read data 
        df_data = ReadData.read_data(self)

        #selection feature
        df_data = df_data.drop(["customer_id"], axis=1)

        print("=========================================")

        #infomation data
        print(df_data.info())

        print("=========================================")

        #check null value
        print("Check Null Value:")
        print(df_data.isnull().any())
        print('=========================================')

        #print mean
        print("Mean:\n", df_data.mean(numeric_only = True))
        print('=========================================')

        #print median
        print("Median:\n", df_data.median(numeric_only = True))
        print('=========================================')

        #print skewness
        print("Skewness:\n", df_data.skew(numeric_only = True))
        print('=========================================')

        #print mode
        print("Mode:\n", df_data.select_dtypes(include = 'object').mode().iloc[0])
        print('=========================================')

        #print descriptive statistic data
        print("Descriptive Statistic:")
        print(df_data.describe().T)
        print('=========================================')

        #=================== Plot ======================
        try:

            #visualization pie chart
            fig_data = plt.figure()

            #add axes ([xmin, ymin, dx, dy])
            axes_data = fig_data.add_axes([0, 0, 1, 1])

            #add labels
            labels_data = ['N', 'Y']
            churn_data = df_data["loan_status"].value_counts()
            axes_data.pie(churn_data, labels=labels_data, autopct='%.0f%%')
            
            #save image
            plt.savefig("./application/chart/pie_chart_dependent_variable.png")

            #clear current fig
            plt.cla()

            #background visualization plot
            sns.set(style='whitegrid')

            #sub plot 3 x 3
            fig, ax = plt.subplots(3, 2, figsize=(16,14))

            #print chart data categorical
            sns.countplot(data=df_data, x='gender', hue='loan_status', ax=ax[0,0])
            sns.countplot(data=df_data, x='married', hue='loan_status', ax=ax[0,1])
            sns.countplot(data=df_data, x='dependents', hue='loan_status', ax=ax[1,0])
            sns.countplot(data=df_data, x='education', hue='loan_status', ax=ax[1,1])
            sns.countplot(data=df_data, x='self_employed', hue='loan_status', ax=ax[2,0])
            sns.countplot(data=df_data, x='property_area', hue='loan_status', ax=ax[2,1])

            plt.tight_layout()

            #save image
            plt.savefig("./application/chart/chart_data_categorical.png")

            #clear current fig
            plt.cla()

            #visualization pie chart
            fig_data = plt.figure()

            #background visualization plot
            sns.set(style='whitegrid')

            #print chart correlation
            sns.heatmap(round(df_data.corr(), 2))

            plt.tight_layout()

            #save image
            plt.savefig("./application/chart/chart_correlation.png")

            #clear current fig
            plt.cla()

            #print status
            print("success save image")
        
        except:

            print("cannot save image")
        
        print("=========================================")

       