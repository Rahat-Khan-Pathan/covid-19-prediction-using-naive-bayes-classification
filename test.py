import pandas as pd
import json
import time

def pre_processing(df):
    print("Processing Test Data...")
    time.sleep(2)
    x = df.drop(df.columns[-1],axis=1)
    y = df[df.columns[-1]]
    print("Done!\n")
    return x, y

def read_data():
    print("Reading Test Data...")
    time.sleep(2)
    df = pd.read_csv("./datasets/covid.csv")
    print("Done!\n")
    return df

def load_model():
    print("Loading Trained Data from trained_data.txt...")
    time.sleep(2)
    with open('trained_data.txt') as f:
        data = f.read()
        js = json.loads(data)
    print("Loaded!\n")
    return js

def test_model(df,x,y,test,testAns,trained_data):
    print(f"Model Testing In Progress. Testing with {len(test)} Data...")
    time.sleep(3)
    totalTrained=len(test)
    correct=0
    total=len(df[y.name])
    totalYes=trained_data["totalYes"]
    totalNo=trained_data["totalNo"]
    for indx in range(len(test)):
        singleRow=test.iloc[indx]
        sum1Yes=(totalYes/total)
        sum2Yes=1
        for indx2 in range(len(singleRow)):
            # print(test.columns[indx2],singleRow[indx2],trained_data[test.columns[indx2]][singleRow[indx2]]['yes'])
            sum1Yes*=(trained_data[test.columns[indx2]][singleRow[indx2]]['yes']/totalYes)
            sum2Yes*=(trained_data[test.columns[indx2]][singleRow[indx2]]['total']/total)
        # print(sum1Yes,sum2Yes)
        sumYes = sum1Yes/sum2Yes
        sum1No=(totalNo/total)
        sum2No=1
        for indx2 in range(len(singleRow)):
            sum1No*=(trained_data[test.columns[indx2]][singleRow[indx2]]['no']/totalNo)
            sum2No*=(trained_data[test.columns[indx2]][singleRow[indx2]]['total']/total)
        sumNo = sum1No/sum2No
        # print(sumYes,sumNo,list(testAns)[indx])
        # print(list(testAns)[indx])
        if sumYes>sumNo:
            ans="Yes"
        else:
            ans="No"
        # print(list(testAns)[indx])
        if ans == list(testAns)[indx]:
            correct+=1
    print("Test Done!")
    print(f"Accuracy - {format((correct/totalTrained)*100,'.2f')}%\n")

df = pd.DataFrame(read_data())
x,y=pre_processing(df)
x=x.truncate(0,4433)
y=y.truncate(0,4433)
test=df.truncate(4434,5433)
testAns=test[test.columns[-1]]
test=test.drop(test.columns[-1],axis=1)
trained_data=load_model()
# print(test,testAns)
test_model(df,x,y,test,testAns,trained_data)
