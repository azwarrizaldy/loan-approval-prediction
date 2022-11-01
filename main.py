"""
    author  : azwar8597@gmail.com
    project : Machine Learning KNN 
"""


from application import (
         TrainingData, EdaData
)
import argparse #type: ignore


if __name__ == "__main__":

    parser      = argparse.ArgumentParser()
    parser.add_argument("--action", help="'eda' / 'train'", type=str)
    args        = parser.parse_args()
    action      = args.action
    if not action:
        while True:
            action = input("'eda' or 'train'?\n")
            if not action:
                print("Please input argument")
            else:
                break
    if action not in['eda', 'train']:
        raise ValueError("invalid Argument: only input eda or train")
    
    if action == 'eda':
        print("Explanatory Data Analysis (EDA)............")
        eda = EdaData()
        eda.eda_data()
    elif action == 'train':
        print("Training K-Nearest Neighbors............")
        trainer = TrainingData()
        trainer.train()
        print("Training Completed and Save Model to S3")