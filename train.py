import pandas as pd
import json
import numpy as np
import time

def read_data():
    print("Reading Data from CSV...")
    time.sleep(2)
    df = pd.read_csv("./datasets/covid.csv")
    print("Done!\n")
    return df

def pre_processing(df):
    print("Processing Data. Splitting Training Data and Testing Data. Taking 1000 Data for Testing..")
    time.sleep(3)
    x = df.drop(df.columns[-1],axis=1)
    y = df[df.columns[-1]]
    print("Done!\n")
    return x, y

def train_model(df,x,y):
    print(f"Training Model with {len(x)} Data...")
    time.sleep(3)
    totalYes=0
    totalNo=0
    allFeatures=list(x.columns)
    trained_data={}
    # time.sleep(2)
    for item in y:
        if item=="Yes":
            totalYes+=1
        else:
            totalNo+=1
    for feature in allFeatures:
        dct={}
        uniqueItemsList = list(df[feature])
        for uniqueItem in np.unique(uniqueItemsList):
            yes=0
            no=0
            for indx in range(len(list(x[feature]))):
                if x[feature][indx]==uniqueItem and df[y.name][indx]=="Yes":
                    yes+=1
                elif x[feature][indx]==uniqueItem:
                    no+=1
            dct[uniqueItem]={"yes":yes,"no":no,"total":yes+no}
        trained_data[feature]=dct
    trained_data["totalYes"]=totalYes
    trained_data["totalNo"]=totalNo
    with open('trained_data.txt', 'w') as convert_file:
        convert_file.write(json.dumps(trained_data))
    print("Model Trained. Model Extracted in trained_data.txt File. Now You Can Test Your Model with Test Data!\n")

df = pd.DataFrame(read_data())
x,y=pre_processing(df)
# print(len(x))
x=x.truncate(0,4433)
y=y.truncate(0,4433)
# print(y)
train_model(df,x,y)