"""
    author  : azwar8597@gmail.com
    project : Machine Learning KNN 
"""


import pandas as pd #type: ignore
import psycopg2 #type: ignore
from sqlalchemy import create_engine #type: ignore
from dotenv import load_dotenv #type: ignore
import os #type: ignore

class ReadData:

    def read_data(self):

        #load function dotenv
        load_dotenv()

        #define env url server RDS 
        URL             = os.environ['url']

        #define env user server RDS 
        USER            = os.environ['usr']

        #define env password server RDS 
        PASSWORD        = os.environ['pas']

        #define env port server RDS 
        PORT            = os.environ['por']

        #define env database
        DATABASE        = os.environ['db']

        #initiation connect to postgresql
        engine = create_engine("postgresql://{USER}:{PASSWORD}@{URL}:{PORT}/{DATABASE}".format(
                                        USER        = USER,
                                        PASSWORD    = PASSWORD,
                                        URL         = URL,
                                        PORT        = PORT,
                                        DATABASE    = DATABASE
                                    )
                                )
    
        #query to get dataset machine learning from postgresql
        query = """
                    WITH customer_property as (
	                    WITH customer_repayment as (
		                    WITH customer_credit as (
		                        SELECT *
                                FROM customer as a
                                INNER JOIN source_income as b
                                ON a.customer_id = b.income_id)
                            SELECT *
                            FROM customer_credit as a
                            INNER JOIN loan_repayment as b
                            ON a.income_id = b.repayment_id)
                        SELECT *
                        FROM customer_repayment as a
                        INNER JOIN property as b
                        ON a.repayment_id = b.property_id)
                    SELECT *
                    FROM customer_property as a
                    INNER JOIN loan_status as b
                    ON a.property_id = b.status_id;
                """

        #read data from postgresql
        df_data = pd.read_sql(query, engine)

        #clean data
        df_data.drop(["income_id", "repayment_id", "property_id", "status_id"], axis=1, inplace=True)
        
        return df_data


    