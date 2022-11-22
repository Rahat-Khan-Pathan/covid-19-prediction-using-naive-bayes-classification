
import pandas as pd
import numpy as np
import time
import json

def pre_processing(df):
    print("Processing Test Data...")
    time.sleep(2)
    x = df.drop(df.columns[-1],axis=1)
    y = df[df.columns[-1]]
    print("Done!\n")
    return x, y

def test_model(df,x,y,testData,trained_data):
    print(f"Model Testing In Progress...")
    time.sleep(3)
    total=len(df[y.name])
    totalYes=trained_data["totalYes"]
    totalNo=trained_data["totalNo"]
    
    sum1Yes=(totalYes/total)
    sum2Yes=1
    for indx2 in range(len(testData)):
        # print(test.columns[indx2],singleRow[indx2],trained_data[test.columns[indx2]][singleRow[indx2]]['yes'])
        sum1Yes*=(trained_data[x.columns[indx2]][testData[indx2]]['yes']/totalYes)
        sum2Yes*=(trained_data[x.columns[indx2]][testData[indx2]]['total']/total)
    # print(sum1Yes,sum2Yes)
    sumYes = sum1Yes/sum2Yes
    sum1No=(totalNo/total)
    sum2No=1
    for indx2 in range(len(testData)):
        sum1No*=(trained_data[x.columns[indx2]][testData[indx2]]['no']/totalNo)
        sum2No*=(trained_data[x.columns[indx2]][testData[indx2]]['total']/total)
    sumNo = sum1No/sum2No
    # print(sumYes,sumNo,list(testAns)[indx])
    # print(list(testAns)[indx])
    print(f"\nProbability of Covid 19 to be True : {format(sumYes,'.2f')}")
    print(f"Probability of Covid 19 to be False : {format(sumNo,'.2f')}")
    if sumYes>sumNo:
        ans="Yes"
    else:
        ans="No"
    print(f"Answer is - {ans}\n\n\n")

def read_data():
    print("Reading data from csv...")
    time.sleep(2)
    df = pd.read_csv("./datasets/covid.csv")
    print("Done...\n")
    return df

def load_model():
    print("Loading Trained Data from trained_data.txt...")
    time.sleep(2)
    with open('./trained/trained_data.txt') as f:
        data = f.read()
        js = json.loads(data)
    print("Loaded!\n")
    return js

df = pd.DataFrame(read_data())
x,y=pre_processing(df)
x=x.truncate(0,4433)
y=y.truncate(0,4433)
trained_data=load_model()

while True:
    print("\nEnter you choice of features:\n")
    test=[]
    for uniqueCol in list(x.columns):
        columnData=list(np.unique(df[uniqueCol]))
        for idx in range(len(columnData)):
            print(f"{idx}. {columnData[idx]}")
        inp=int(input(f"Enter option for {uniqueCol}: "))
        test.append(columnData[inp])
        print("\n")
    print(f"Your test data is: {test}\n")
    test_model(df,x,y,test,trained_data)
    

            

