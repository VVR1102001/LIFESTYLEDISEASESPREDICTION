import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
from flask import Flask, render_template, request, redirect, Response
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns

import time
import random as r

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier


import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense




app = Flask(__name__)
app.secret_key = 'dropboxapp1234'
global cls1, cls2, cls3, cls4, cls5, cls6, cls7, cls8, cls9,model,classifier,MLP,RF, accmod,X,KNN,GBC,LGBM,ETC,mmodel
#global classifier1,classifier2,classifier3,classifier5,classifier6
 


accmod={}

dapath = "Dataset/COMG09N.csv"

df = pd.read_csv(dapath)
columnss = df.columns
    
#norm7 = normalize([df[df.columns[7]]])
norm1 = normalize([df[df.columns[1]]])
norm2 = normalize([df[df.columns[2]]])
#norm8 = normalize([df[df.columns[8]]])
norm11 = normalize([df[df.columns[11]]])
#norm5 = normalize([df[df.columns[5]]])
    
#df[df.columns[7]]=norm7[0][0:]
df[df.columns[1]]=norm1[0][0:]
df[df.columns[2]]=norm2[0][0:]
df[df.columns[11]]=norm11[0][0:]
#df[df.columns[8]]=norm8[0][0:]
#df[df.columns[5]]=norm5[0][0:]

le = LabelEncoder()

columnss = df.columns
df[columnss[0]]= le.fit_transform(df[columnss[0]])

df[columnss[3]]= le.fit_transform(df[columnss[3]])

df[columnss[4]]= le.fit_transform(df[columnss[4]])
df[columnss[7]]= le.fit_transform(df[columnss[7]])
df[columnss[8]]= le.fit_transform(df[columnss[8]])

df[columnss[9]]= le.fit_transform(df[columnss[9]])

df[columnss[10]]= le.fit_transform(df[columnss[10]])

#df[columnss[11]]= le.fit_transform(df[columnss[11]])

df[columnss[12]]= le.fit_transform(df[columnss[12]])
df[columnss[13]]= le.fit_transform(df[columnss[13]])

#df[columnss[19]]= le.fit_transform(df[columnss[19]])
df[columnss[14]]= le.fit_transform(df[columnss[14]])
df[columnss[15]]= le.fit_transform(df[columnss[15]])
df[columnss[16]]= le.fit_transform(df[columnss[16]])

df[columnss[17]]= le.fit_transform(df[columnss[17]])
#df[columnss[18]]= le.fit_transform(df[columnss[18]])



    

#print(df[columnss[0:14]])
    
    
X=df[df.columns[0:12]]
Y=df["HeartDisease"].values

Y1=df["Asthma"].values
Y2=df["KidneyDisease"].values
Y3=df["SkinCancer"].values
Y5=df["Stroke"].values
Y6=df["HeartDisease"].values

      
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

X_train1, X_test1,Y_train1,Y_test1 = train_test_split(X,Y1,test_size=0.2,random_state=2)
X_train2, X_test2,Y_train2,Y_test2 = train_test_split(X,Y2,test_size=0.2,random_state=2)
X_train3, X_test3,Y_train3,Y_test3 = train_test_split(X,Y3,test_size=0.2,random_state=2)
X_train5, X_test5,Y_train5,Y_test5 = train_test_split(X,Y5,test_size=0.2,random_state=2)
X_train6, X_test6,Y_train6,Y_test6 = train_test_split(X,Y6,test_size=0.2,random_state=2)


    
sc = StandardScaler()
sc.fit(X_train)
#X_train = sc.transform(X_train)
#X_test = sc.transform(X_test)



@app.route("/Visualize")
def Visualize():
    df = pd.read_csv(dapath)
    oe = 0
    et = 0
    for i in df["Fruits"].values:
        if i==1:
            oe+=1
        elif i == 0:
            et+=1
    plt.title("Fruits Eating Graph")
    plt.bar(np.array(["Fruits-Yes","Fruits-No"]),np.array([oe,et]),color=["yellow","pink"])
    plt.show()


    time.sleep(2)

    os = 0
    cs = 0

    for i in df["Smoking"].values:
        if i=="Yes":
            os+=1
        elif i == "No":
            cs+=1

        

    plt.title("Smokers Graph")
    plt.bar(np.array(["Smoking","No-Smoking"]),np.array([os,cs]),color=["orange","black"])
    plt.show()

    time.sleep(2)

    so = 0
    ad = 0
    for i in df["AlcoholDrinking"].values:
        if i=="Yes":
            so+=1
        elif i == "No":
            ad+=1
        
    plt.title("Drinkers Graph")
    plt.bar(np.array(["Addicted Drinker","Non-Drinker"]),np.array([so,ad]),color=["blue","red"])

    plt.show()

   
    sag = pd.DataFrame({"Smoking":df["Smoking"],"Age":df["Age"]})
    sag = sag[sag.Smoking!="No"]
  
    aag = pd.DataFrame({"AlcoholDrinking":df["AlcoholDrinking"],"Age":df["Age"]})
    aag = aag[aag.AlcoholDrinking!="No"]

    sag.groupby('Smoking').Age.plot()
  
    sns.barplot(x="Smoking", y="Age",data=sag)
    plt.show()
   

    aag.groupby('AlcoholDrinking').Age.plot()

    sns.barplot(x="AlcoholDrinking", y="Age",data=aag)
    plt.show() 

  
    output = ""
    return render_template("AdminScreen.html",error=output)


    

@app.route("/TrainModel")
def TrainModel():
    output = "<center><h4>Train Model</h4><a href='TrainSVM'>Train SVM</a><br>"
    output+="<br><a href='TrainMLP'>Train MLP</a><br>"
    output+="<br><a href='TrainKNN'>Train KNN</a><br>"
    output+="<br><a href='TrainETC'>Train ExtraTreesClassifier</a><br>"
    output+="<br><a href='TrainGB'>Train GBC</a><br>"
    output+="<br><a href='TrainLGBM'>Train LGBM</a><br>"
    output+="<br><a href='TrainRF'>Train RF</a><br>"
    output+="<br><a href='TrainCNN'>Train CNN</a><br>"
    output+="<br><a href='TrainMCNN'>Train MCNN</a></center><br>"
    
    
    return render_template("ViewAccuracy.html",error=output)
	

@app.route("/TrainSVM")
def TrainSVM():
    global classifier,accmod,svm_teacc,svm_tracc,classifier1,classifier2,classifier3,classifier5,classifier6
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17
    
    classifier = svm.SVC()

    classifier.fit(X_train,Y_train)
    
    classifier1 = svm.SVC()
    classifier2 = svm.SVC()
    classifier3 = svm.SVC()
    #classifier4 = svm.SVC()
    classifier5 = svm.SVC()
    classifier6 = svm.SVC()
    
    classifier1.fit(X_train1,Y_train1)
    classifier2.fit(X_train2,Y_train2)
    classifier3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    classifier5.fit(X_train5,Y_train5)
    classifier6.fit(X_train6,Y_train6)



    svmtrpred = classifier.predict(X_train)
    svmtr_acc = accuracy_score(svmtrpred,Y_train)
    print('Accuracy on training data : ', svmtr_acc*100)
    
    
    svmtrpred1 = classifier1.predict(X_train1)
    svmtr_acc1 = accuracy_score(svmtrpred1,Y_train1)
    print('Accuracy on training data : ', svmtr_acc1*100)
    
    
    svmtrpred2 = classifier2.predict(X_train2)
    svmtr_acc2 = accuracy_score(svmtrpred2,Y_train2)
    print('Accuracy on training data : ', svmtr_acc2*100)
    
    svmtrpred3 = classifier3.predict(X_train3)
    svmtr_acc3 = accuracy_score(svmtrpred3,Y_train3)
    print('Accuracy on training data : ', svmtr_acc3*100)
    
    svmtrpred5 = classifier5.predict(X_train5)
    svmtr_acc5 = accuracy_score(svmtrpred5,Y_train5)
    print('Accuracy on training data : ', svmtr_acc5*100)
    
    svmtrpred6 = classifier6.predict(X_train6)
    svmtr_acc6 = accuracy_score(svmtrpred6,Y_train6)
    print('Accuracy on training data : ', svmtr_acc6*100)
    
    svmtepred1 = classifier1.predict(X_test1)
    svmte_acc1 = accuracy_score(svmtepred1,Y_test1)


    print('Accuracy on test data : ', svmte_acc1*100)


    svmtepred2 = classifier2.predict(X_test2)
    svmte_acc2 = accuracy_score(svmtepred2,Y_test2)


    print('Accuracy on test data : ', svmte_acc2*100)


    svmtepred3= classifier3.predict(X_test3)
    svmte_acc3 = accuracy_score(svmtepred3,Y_test3)


    print('Accuracy on test data : ', svmte_acc3*100)


    svmtepred5 = classifier5.predict(X_test5)
    svmte_acc5 = accuracy_score(svmtepred5,Y_test5)


    print('Accuracy on test data : ', svmte_acc5*100)


    svmtepred6 = classifier6.predict(X_test6)
    svmte_acc6 = accuracy_score(svmtepred6,Y_test6)


    print('Accuracy on test data : ', svmte_acc6*100)
    
    svm_teacc1 = svmte_acc1*100
    svm_teacc2 = svmte_acc1*100
    svm_teacc3 = svmte_acc1*100
    svm_teacc5 = svmte_acc1*100
    svm_teacc6 = svmte_acc1*100
    
    svm_tracc1 = svmtr_acc1*100
    svm_tracc2 = svmtr_acc1*100
    svm_tracc3 = svmtr_acc1*100
    svm_tracc5 = svmtr_acc1*100
    svm_tracc6 = svmtr_acc1*100


    #svmtepred = classifier.predict(X_test)
    #svmte_acc = accuracy_score(svmtepred,Y_test)


    #print('Accuracy on testing data : ', svmte_acc*100)
    #accmod.append(svmte_acc*100)
    #
    
    num_list = [svm_teacc1,svm_teacc2,svm_teacc3,svm_teacc5,svm_teacc6]
    svm_teacc = sum(num_list)/len(num_list)
    
    num_list1 = [svm_tracc1,svm_tracc2,svm_tracc3,svm_tracc5,svm_tracc6]
    svm_tracc = sum(num_list1)/len(num_list1)

    
    svmtepred_list = [svmtepred1,svmtepred2,svmtepred3,svmtepred5,svmtepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    svmtepred = svmtepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["SVM"] = svm_teacc
    
    
    
    pre = precision_score(Y_test,svmtepred,average=None)
    recall = recall_score(Y_test,svmtepred,average=None)
    f1 = f1_score(Y_test,svmtepred,average=None)
    
    output = svm_teacc
    color = '<font size="" color="black">'
    output = '<center><h2>SVM</h2><br><br><table border="1" align="center">'
    output+='<tr><th>SVM Training Accuracy</th><th>SVM Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(svm_tracc)+'</td><td>'+color+str(svm_teacc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,svmtepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("SVM Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = svm_teacc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("SVM - Precision - Recall - F1 Measure")
    plt.show()
	
    return render_template("ViewAccuracy.html",error=output)
    
    
@app.route("/TrainKNN")
def TrainKNN():
    global KNN,accmod,knn_teacc,knn_tracc,KNN1,KNN2,KNN3,KNN5,KNN6
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17
    
    KNN = KNeighborsClassifier()
    KNN.fit(X_train,Y_train)
    #knntrpred = KNN.predict(X_train)
    #knntr_acc = accuracy_score(knntrpred,Y_train)
    #print('Accuracy on training data : ', knntr_acc*100)
    
    #knntepred = KNN.predict(X_test)
    #knnte_acc = accuracy_score(knntepred,Y_test)
    #print('Accuracy on testing data : ', knnte_acc*100)
    
    
    KNN1 = KNeighborsClassifier()
    KNN2 = KNeighborsClassifier()
    KNN3 =KNeighborsClassifier()
    #classifier4 = svm.SVC()
    KNN5 = KNeighborsClassifier()
    KNN6 = KNeighborsClassifier()
    
    KNN1.fit(X_train1,Y_train1)
    KNN2.fit(X_train2,Y_train2)
    KNN3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    KNN5.fit(X_train5,Y_train5)
    KNN6.fit(X_train6,Y_train6)



    
    
    
    trpred1 = KNN1.predict(X_train1)
    tr_acc1 = accuracy_score(trpred1,Y_train1)
    print('Accuracy on training data : ', tr_acc1*100)
    
    
    trpred2 = KNN2.predict(X_train2)
    tr_acc2 = accuracy_score(trpred2,Y_train2)
    print('Accuracy on training data : ', tr_acc2*100)
    
    trpred3 = KNN3.predict(X_train3)
    tr_acc3 = accuracy_score(trpred3,Y_train3)
    print('Accuracy on training data : ', tr_acc3*100)
    
    trpred5 = KNN5.predict(X_train5)
    tr_acc5 = accuracy_score(trpred5,Y_train5)
    print('Accuracy on training data : ', tr_acc5*100)
    
    trpred6 = KNN6.predict(X_train6)
    tr_acc6 = accuracy_score(trpred6,Y_train6)
    print('Accuracy on training data : ', tr_acc6*100)
    
    tepred1 = KNN1.predict(X_test1)
    te_acc1 = accuracy_score(tepred1,Y_test1)


    print('Accuracy on test data : ', te_acc1*100)


    tepred2 = KNN2.predict(X_test2)
    te_acc2 = accuracy_score(tepred2,Y_test2)


    print('Accuracy on test data : ', te_acc2*100)


    tepred3= KNN3.predict(X_test3)
    te_acc3 = accuracy_score(tepred3,Y_test3)


    print('Accuracy on test data : ', te_acc3*100)


    tepred5 = KNN5.predict(X_test5)
    te_acc5 = accuracy_score(tepred5,Y_test5)


    print('Accuracy on test data : ', te_acc5*100)


    tepred6 = KNN6.predict(X_test6)
    te_acc6 = accuracy_score(tepred6,Y_test6)


    print('Accuracy on test data : ', te_acc6*100)
    
    teacc1 = te_acc1*100
    teacc2 = te_acc2*100
    teacc3 = te_acc3*100
    teacc5 = te_acc5*100
    teacc6 = te_acc6*100
    
    tracc1 = tr_acc1*100
    tracc2 = tr_acc2*100
    tracc3 = tr_acc3*100
    tracc5 = tr_acc5*100
    tracc6 = tr_acc6*100


    
    
    num_list = [teacc1,teacc2,teacc3,teacc5,teacc6]
    teaccc = sum(num_list)/len(num_list)
    
    num_list1 = [tracc1,tracc2,tracc3,tracc5,tracc6]
    traccc = sum(num_list1)/len(num_list1)

    
    tepred_list = [tepred1,tepred2,tepred3,tepred5,tepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    tepred = tepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["KNN"] = teaccc
    knn_teacc = teaccc
    knn_tracc = tracc
    
    
    
    
    pre = precision_score(Y_test,tepred,average=None)
    recall = recall_score(Y_test,tepred,average=None)
    f1 = f1_score(Y_test,tepred,average=None)
    output = teaccc
    color = '<font size="" color="black">'
    output = '<center><h2>KNN</h2><br><br><table border="1" align="center">'
    output+='<tr><th>KNN Training Accuracy</th><th>KNN Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(traccc)+'</td><td>'+color+str(teaccc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,tepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("KNN Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = teaccc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("KNN - Precision - Recall - F1 Measure")
    plt.show()
    
    
    return render_template("ViewAccuracy.html",error=output)
    
    
@app.route("/TrainGB")
def TrainGB():
    global GBC,accmod,gb_teacc,gb_tracc,GBC1,GBC2,GBC3,GBC5,GBC6
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17
    
    GBC = GradientBoostingClassifier()
    GBC.fit(X_train,Y_train)
    gbtrpred = GBC.predict(X_train)
    gbtr_acc = accuracy_score(gbtrpred,Y_train)
    print('Accuracy on training data : ', gbtr_acc*100)
    
    #gbtepred = GBC.predict(X_test)
    #gbte_acc = accuracy_score(gbtepred,Y_test)
    #print('Accuracy on testing data : ', gbte_acc*100)
    
    GBC1 = GradientBoostingClassifier()
    GBC2 = GradientBoostingClassifier()
    GBC3 =GradientBoostingClassifier()
    #classifier4 = svm.SVC()
    GBC5 = GradientBoostingClassifier()
    GBC6 = GradientBoostingClassifier()
    
    GBC1.fit(X_train1,Y_train1)
    GBC2.fit(X_train2,Y_train2)
    GBC3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    GBC5.fit(X_train5,Y_train5)
    GBC6.fit(X_train6,Y_train6)



    
    
    
    trpred1 = GBC1.predict(X_train1)
    tr_acc1 = accuracy_score(trpred1,Y_train1)
    print('Accuracy on training data : ', tr_acc1*100)
    
    
    trpred2 = GBC2.predict(X_train2)
    tr_acc2 = accuracy_score(trpred2,Y_train2)
    print('Accuracy on training data : ', tr_acc2*100)
    
    trpred3 = GBC3.predict(X_train3)
    tr_acc3 = accuracy_score(trpred3,Y_train3)
    print('Accuracy on training data : ', tr_acc3*100)
    
    trpred5 = GBC5.predict(X_train5)
    tr_acc5 = accuracy_score(trpred5,Y_train5)
    print('Accuracy on training data : ', tr_acc5*100)
    
    trpred6 = GBC6.predict(X_train6)
    tr_acc6 = accuracy_score(trpred6,Y_train6)
    print('Accuracy on training data : ', tr_acc6*100)
    
    tepred1 = GBC1.predict(X_test1)
    te_acc1 = accuracy_score(tepred1,Y_test1)


    print('Accuracy on test data : ', te_acc1*100)


    tepred2 = GBC2.predict(X_test2)
    te_acc2 = accuracy_score(tepred2,Y_test2)


    print('Accuracy on test data : ', te_acc2*100)


    tepred3= GBC3.predict(X_test3)
    te_acc3 = accuracy_score(tepred3,Y_test3)


    print('Accuracy on test data : ', te_acc3*100)


    tepred5 = GBC5.predict(X_test5)
    te_acc5 = accuracy_score(tepred5,Y_test5)


    print('Accuracy on test data : ', te_acc5*100)


    tepred6 = GBC6.predict(X_test6)
    te_acc6 = accuracy_score(tepred6,Y_test6)


    print('Accuracy on test data : ', te_acc6*100)
    
    teacc1 = te_acc1*100
    teacc2 = te_acc2*100
    teacc3 = te_acc3*100
    teacc5 = te_acc5*100
    teacc6 = te_acc6*100
    
    tracc1 = tr_acc1*100
    tracc2 = tr_acc2*100
    tracc3 = tr_acc3*100
    tracc5 = tr_acc5*100
    tracc6 = tr_acc6*100

    
    num_list = [teacc1,teacc2,teacc3,teacc5,teacc6]
    teaccc = sum(num_list)/len(num_list)
    
    num_list1 = [tracc1,tracc2,tracc3,tracc5,tracc6]
    traccc = sum(num_list1)/len(num_list1)

    
    tepred_list = [tepred1,tepred2,tepred3,tepred5,tepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    tepred = tepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["GBC"] = teaccc
    gb_teacc = teaccc
    gb_tracc=tracc
    
    
    
    
    pre = precision_score(Y_test,tepred,average=None)
    recall = recall_score(Y_test,tepred,average=None)
    f1 = f1_score(Y_test,tepred,average=None)
    output = teaccc
    color = '<font size="" color="black">'
    output = '<center><h2>GBC</h2><br><br><table border="1" align="center">'
    output+='<tr><th>GBC Training Accuracy</th><th>GBC Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(traccc)+'</td><td>'+color+str(teaccc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,tepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("GBC Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = teaccc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("GBC - Precision - Recall - F1 Measure")
    plt.show()
    
    
    return render_template("ViewAccuracy.html",error=output)


@app.route("/TrainETC")
def TrainETC():
    global ETC,accmod,etc_teacc,etc_tracc,ETC1,ETC2,ETC3,ETC5,ETC6
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17
    
    ETC = ExtraTreesClassifier()
    ETC.fit(X_train,Y_train)
    etctrpred = ETC.predict(X_train)
    etctr_acc = accuracy_score(etctrpred,Y_train)
    print('Accuracy on training data : ', etctr_acc*100)
    
    #etctepred = ETC.predict(X_test)
    #etcte_acc = accuracy_score(etctepred,Y_test)
    #print('Accuracy on testing data : ', etcte_acc*100)
    
    
    ETC1 = ExtraTreesClassifier()
    ETC2 = ExtraTreesClassifier()
    ETC3 =ExtraTreesClassifier()
    #classifier4 = svm.SVC()
    ETC5 = ExtraTreesClassifier()
    ETC6 = ExtraTreesClassifier()
    
    ETC1.fit(X_train1,Y_train1)
    ETC2.fit(X_train2,Y_train2)
    ETC3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    ETC5.fit(X_train5,Y_train5)
    ETC6.fit(X_train6,Y_train6)



    
    
    
    trpred1 = ETC1.predict(X_train1)
    tr_acc1 = accuracy_score(trpred1,Y_train1)
    print('Accuracy on training data : ', tr_acc1*100)
    
    
    trpred2 = ETC2.predict(X_train2)
    tr_acc2 = accuracy_score(trpred2,Y_train2)
    print('Accuracy on training data : ', tr_acc2*100)
    
    trpred3 = ETC3.predict(X_train3)
    tr_acc3 = accuracy_score(trpred3,Y_train3)
    print('Accuracy on training data : ', tr_acc3*100)
    
    trpred5 = ETC5.predict(X_train5)
    tr_acc5 = accuracy_score(trpred5,Y_train5)
    print('Accuracy on training data : ', tr_acc5*100)
    
    trpred6 = ETC6.predict(X_train6)
    tr_acc6 = accuracy_score(trpred6,Y_train6)
    print('Accuracy on training data : ', tr_acc6*100)
    
    tepred1 = ETC1.predict(X_test1)
    te_acc1 = accuracy_score(tepred1,Y_test1)


    print('Accuracy on test data : ', te_acc1*100)


    tepred2 = ETC2.predict(X_test2)
    te_acc2 = accuracy_score(tepred2,Y_test2)


    print('Accuracy on test data : ', te_acc2*100)


    tepred3= ETC3.predict(X_test3)
    te_acc3 = accuracy_score(tepred3,Y_test3)


    print('Accuracy on test data : ', te_acc3*100)


    tepred5 = ETC5.predict(X_test5)
    te_acc5 = accuracy_score(tepred5,Y_test5)


    print('Accuracy on test data : ', te_acc5*100)


    tepred6 = ETC6.predict(X_test6)
    te_acc6 = accuracy_score(tepred6,Y_test6)


    print('Accuracy on test data : ', te_acc6*100)
    
    teacc1 = te_acc1*100
    teacc2 = te_acc2*100
    teacc3 = te_acc3*100
    teacc5 = te_acc5*100
    teacc6 = te_acc6*100
    
    tracc1 = tr_acc1*100
    tracc2 = tr_acc2*100
    tracc3 = tr_acc3*100
    tracc5 = tr_acc5*100
    tracc6 = tr_acc6*100

    
    num_list = [teacc1,teacc2,teacc3,teacc5,teacc6]
    teaccc = sum(num_list)/len(num_list)
    
    num_list1 = [tracc1,tracc2,tracc3,tracc5,tracc6]
    traccc = sum(num_list1)/len(num_list1)

    
    tepred_list = [tepred1,tepred2,tepred3,tepred5,tepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    tepred = tepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["ETC"] = teaccc
    etc_teacc = teaccc
    etc_tracc=tracc
    
    
    
    pre = precision_score(Y_test,tepred,average=None)
    recall = recall_score(Y_test,tepred,average=None)
    f1 = f1_score(Y_test,tepred,average=None)
    output = teaccc
    color = '<font size="" color="black">'
    output = '<center><h2>ETC</h2><br><br><table border="1" align="center">'
    output+='<tr><th>ETC Training Accuracy</th><th>ETC Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(traccc)+'</td><td>'+color+str(teaccc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,tepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("ETC Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = teaccc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("ETC - Precision - Recall - F1 Measure")
    plt.show()
    
    
    return render_template("ViewAccuracy.html",error=output)


@app.route("/TrainLGBM")
def TrainLGBM():
    global LGBM,accmod,lgbm_teacc,LGBM1,LGBM2,LGBM3,LGBM5,LGBM6,lgbm_tracc
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17
    
    LGBM = LGBMClassifier()
    LGBM.fit(X_train,Y_train)
    lgbmtrpred = LGBM.predict(X_train)
    lgbmtr_acc = accuracy_score(lgbmtrpred,Y_train)
    print('Accuracy on training data : ', lgbmtr_acc*100)
    
    #lgbmtepred = LGBM.predict(X_test)
    #lgbmte_acc = accuracy_score(lgbmtepred,Y_test)
    #print('Accuracy on testing data : ', lgbmte_acc*100)
    
    LGBM1 = LGBMClassifier()
    LGBM2 = LGBMClassifier()
    LGBM3 = LGBMClassifier()
    #classifier4 = svm.SVC()
    LGBM5 = LGBMClassifier()
    LGBM6 = LGBMClassifier()
    
    LGBM1.fit(X_train1,Y_train1)
    LGBM2.fit(X_train2,Y_train2)
    LGBM3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    LGBM5.fit(X_train5,Y_train5)
    LGBM6.fit(X_train6,Y_train6)



    
    
    
    trpred1 = LGBM1.predict(X_train1)
    tr_acc1 = accuracy_score(trpred1,Y_train1)
    print('Accuracy on training data : ', tr_acc1*100)
    
    
    trpred2 = LGBM2.predict(X_train2)
    tr_acc2 = accuracy_score(trpred2,Y_train2)
    print('Accuracy on training data : ', tr_acc2*100)
    
    trpred3 = LGBM3.predict(X_train3)
    tr_acc3 = accuracy_score(trpred3,Y_train3)
    print('Accuracy on training data : ', tr_acc3*100)
    
    trpred5 = LGBM5.predict(X_train5)
    tr_acc5 = accuracy_score(trpred5,Y_train5)
    print('Accuracy on training data : ', tr_acc5*100)
    
    trpred6 = LGBM6.predict(X_train6)
    tr_acc6 = accuracy_score(trpred6,Y_train6)
    print('Accuracy on training data : ', tr_acc6*100)
    
    tepred1 = LGBM1.predict(X_test1)
    te_acc1 = accuracy_score(tepred1,Y_test1)


    print('Accuracy on test data : ', te_acc1*100)


    tepred2 = LGBM2.predict(X_test2)
    te_acc2 = accuracy_score(tepred2,Y_test2)


    print('Accuracy on test data : ', te_acc2*100)


    tepred3= LGBM3.predict(X_test3)
    te_acc3 = accuracy_score(tepred3,Y_test3)


    print('Accuracy on test data : ', te_acc3*100)


    tepred5 = LGBM5.predict(X_test5)
    te_acc5 = accuracy_score(tepred5,Y_test5)


    print('Accuracy on test data : ', te_acc5*100)


    tepred6 = LGBM6.predict(X_test6)
    te_acc6 = accuracy_score(tepred6,Y_test6)


    print('Accuracy on test data : ', te_acc6*100)
    
    teacc1 = te_acc1*100
    teacc2 = te_acc2*100
    teacc3 = te_acc3*100
    teacc5 = te_acc5*100
    teacc6 = te_acc6*100
    
    tracc1 = tr_acc1*100
    tracc2 = tr_acc2*100
    tracc3 = tr_acc3*100
    tracc5 = tr_acc5*100
    tracc6 = tr_acc6*100

    
    num_list = [teacc1,teacc2,teacc3,teacc5,teacc6]
    teaccc = sum(num_list)/len(num_list)
    
    num_list1 = [tracc1,tracc2,tracc3,tracc5,tracc6]
    traccc = sum(num_list1)/len(num_list1)

    
    tepred_list = [tepred1,tepred2,tepred3,tepred5,tepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    tepred = tepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["LGBM"] = teaccc
    lgbm_teacc = teaccc
    lgbm_tracc=tracc
    
    
    
    
    pre = precision_score(Y_test,tepred,average=None)
    recall = recall_score(Y_test,tepred,average=None)
    f1 = f1_score(Y_test,tepred,average=None)
    output = teaccc
    color = '<font size="" color="black">'
    output = '<center><h2>LGBM</h2><br><br><table border="1" align="center">'
    output+='<tr><th>LGBM Training Accuracy</th><th>LGBM Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(traccc)+'</td><td>'+color+str(teaccc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,tepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("LGBM Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = teaccc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("LGBM - Precision - Recall - F1 Measure")
    plt.show()
    
    
    return render_template("ViewAccuracy.html",error=output)


@app.route("/TrainMLP")
def TrainMLP():
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17, MLP,accmod,mlp_teacc,MLP1,MLP2,MLP3,MLP5,MLP6,mlp_tracc
    
 
    #MLP = MLPClassifier(random_state=1, max_iter=1000) 
    #MLP.fit(X,Y)
    #mlptrpred = MLP.predict(X_train) 
    #mlptr_acc = accuracy_score(Y_train,mlptrpred)*100
    #print(mlptr_acc)
	
	



    #mlptepred = MLP.predict(X_test)
    #mlpte_acc = accuracy_score(mlptepred,Y_test)


    #print('Accuracy on test data : ', mlpte_acc*100)
    
    
    MLP1 = MLPClassifier(random_state=1, max_iter=1000)
    MLP2 = MLPClassifier(random_state=1, max_iter=1000)
    MLP3 =MLPClassifier(random_state=1, max_iter=1000)
    #classifier4 = svm.SVC()
    MLP5 = MLPClassifier(random_state=1, max_iter=1000)
    MLP6 = MLPClassifier(random_state=1, max_iter=1000)
    
    MLP1.fit(X_train1,Y_train1)
    MLP2.fit(X_train2,Y_train2)
    MLP3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    MLP5.fit(X_train5,Y_train5)
    MLP6.fit(X_train6,Y_train6)



    
    
    
    trpred1 = MLP1.predict(X_train1)
    tr_acc1 = accuracy_score(trpred1,Y_train1)
    print('Accuracy on training data : ', tr_acc1*100)
    
    
    trpred2 = MLP2.predict(X_train2)
    tr_acc2 = accuracy_score(trpred2,Y_train2)
    print('Accuracy on training data : ', tr_acc2*100)
    
    trpred3 = MLP3.predict(X_train3)
    tr_acc3 = accuracy_score(trpred3,Y_train3)
    print('Accuracy on training data : ', tr_acc3*100)
    
    trpred5 = MLP5.predict(X_train5)
    tr_acc5 = accuracy_score(trpred5,Y_train5)
    print('Accuracy on training data : ', tr_acc5*100)
    
    trpred6 = MLP6.predict(X_train6)
    tr_acc6 = accuracy_score(trpred6,Y_train6)
    print('Accuracy on training data : ', tr_acc6*100)
    
    tepred1 = MLP1.predict(X_test1)
    te_acc1 = accuracy_score(tepred1,Y_test1)


    print('Accuracy on test data : ', te_acc1*100)


    tepred2 = MLP2.predict(X_test2)
    te_acc2 = accuracy_score(tepred2,Y_test2)


    print('Accuracy on test data : ', te_acc2*100)


    tepred3= MLP3.predict(X_test3)
    te_acc3 = accuracy_score(tepred3,Y_test3)


    print('Accuracy on test data : ', te_acc3*100)


    tepred5 = MLP5.predict(X_test5)
    te_acc5 = accuracy_score(tepred5,Y_test5)


    print('Accuracy on test data : ', te_acc5*100)


    tepred6 = MLP6.predict(X_test6)
    te_acc6 = accuracy_score(tepred6,Y_test6)


    print('Accuracy on test data : ', te_acc6*100)
    
    teacc1 = te_acc1*100
    teacc2 = te_acc2*100
    teacc3 = te_acc3*100
    teacc5 = te_acc5*100
    teacc6 = te_acc6*100
    
    tracc1 = tr_acc1*100
    tracc2 = tr_acc2*100
    tracc3 = tr_acc3*100
    tracc5 = tr_acc5*100
    tracc6 = tr_acc6*100

    
    num_list = [teacc1,teacc2,teacc3,teacc5,teacc6]
    teaccc = sum(num_list)/len(num_list)
    
    num_list1 = [tracc1,tracc2,tracc3,tracc5,tracc6]
    traccc = sum(num_list1)/len(num_list1)

    
    tepred_list = [tepred1,tepred2,tepred3,tepred5,tepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    tepred = tepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["ETC"] = teaccc
    mlp_teacc = teaccc
    mlp_tracc=tracc
    
    
    
    
    pre = precision_score(Y_test,tepred,average=None)
    recall = recall_score(Y_test,tepred,average=None)
    f1 = f1_score(Y_test,tepred,average=None)
    output = teaccc
    color = '<font size="" color="black">'
    output = '<center><h2>MLP</h2><br><br><table border="1" align="center">'
    output+='<tr><th>MLP Training Accuracy</th><th>MLP Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(traccc)+'</td><td>'+color+str(teaccc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,tepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("MLP Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = teaccc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("MLP - Precision - Recall - F1 Measure")
    plt.show()
    
    
    return render_template("ViewAccuracy.html",error=output)
    
    
    
    

	
@app.route('/TrainRF')
def TrainRF():
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17,RF,accmod,rf_teacc,RF1,RF2,RF3,RF5,RF6,rf_tracc
 
    RF = RandomForestClassifier() 
    RF.fit(X,Y)
    rftrpred = RF.predict(X_train) 
    rftr_acc = accuracy_score(Y_train,rftrpred)*100
    print(rftr_acc)
	
	



    #rftepred = RF.predict(X_test)
    #rfte_acc = accuracy_score(rftepred,Y_test)


    #print('Accuracy on test data : ', rfte_acc*100)
    
    RF1 = RandomForestClassifier()
    RF2 = RandomForestClassifier()
    RF3 = RandomForestClassifier()
    RF5 = RandomForestClassifier()
    RF6 = RandomForestClassifier()
    
    RF1.fit(X_train1,Y_train1)
    RF2.fit(X_train2,Y_train2)
    RF3.fit(X_train3,Y_train3)
    #classifier4.fit(X_train4,Y_train4)
    RF5.fit(X_train5,Y_train5)
    RF6.fit(X_train6,Y_train6)



    
    
    
    trpred1 = RF1.predict(X_train1)
    tr_acc1 = accuracy_score(trpred1,Y_train1)
    print('Accuracy on training data : ', tr_acc1*100)
    
    
    trpred2 = RF2.predict(X_train2)
    tr_acc2 = accuracy_score(trpred2,Y_train2)
    print('Accuracy on training data : ', tr_acc2*100)
    
    trpred3 = RF3.predict(X_train3)
    tr_acc3 = accuracy_score(trpred3,Y_train3)
    print('Accuracy on training data : ', tr_acc3*100)
    
    trpred5 = RF5.predict(X_train5)
    tr_acc5 = accuracy_score(trpred5,Y_train5)
    print('Accuracy on training data : ', tr_acc5*100)
    
    trpred6 = RF6.predict(X_train6)
    tr_acc6 = accuracy_score(trpred6,Y_train6)
    print('Accuracy on training data : ', tr_acc6*100)
    
    tepred1 = RF1.predict(X_test1)
    te_acc1 = accuracy_score(tepred1,Y_test1)


    print('Accuracy on test data : ', te_acc1*100)


    tepred2 = RF2.predict(X_test2)
    te_acc2 = accuracy_score(tepred2,Y_test2)


    print('Accuracy on test data : ', te_acc2*100)


    tepred3= RF3.predict(X_test3)
    te_acc3 = accuracy_score(tepred3,Y_test3)


    print('Accuracy on test data : ', te_acc3*100)


    tepred5 = RF5.predict(X_test5)
    te_acc5 = accuracy_score(tepred5,Y_test5)


    print('Accuracy on test data : ', te_acc5*100)


    tepred6 = RF6.predict(X_test6)
    te_acc6 = accuracy_score(tepred6,Y_test6)


    print('Accuracy on test data : ', te_acc6*100)
    
    teacc1 = te_acc1*100
    teacc2 = te_acc2*100
    teacc3 = te_acc3*100
    teacc5 = te_acc5*100
    teacc6 = te_acc6*100
    
    tracc1 = tr_acc1*100
    tracc2 = tr_acc2*100
    tracc3 = tr_acc3*100
    tracc5 = tr_acc5*100
    tracc6 = tr_acc6*100

    
    num_list = [teacc1,teacc2,teacc3,teacc5,teacc6]
    teaccc = sum(num_list)/len(num_list)
    
    num_list1 = [tracc1,tracc2,tracc3,tracc5,tracc6]
    traccc = sum(num_list1)/len(num_list1)

    
    tepred_list = [tepred1,tepred2,tepred3,tepred5,tepred6]
    Y_testlist  = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]    
    ra = r.randint(0,4)
    tepred = tepred_list[ra]
    Y_test = Y_testlist[ra]
    
    accmod["RF"] = teaccc
    rf_teacc = teaccc
    rf_tracc=tracc
    
    
    
    
    pre = precision_score(Y_test,tepred,average=None)
    recall = recall_score(Y_test,tepred,average=None)
    f1 = f1_score(Y_test,tepred,average=None)
    output = teaccc
    color = '<font size="" color="black">'
    output = '<center><h2>RF</h2><br><br><table border="1" align="center">'
    output+='<tr><th>RF Training Accuracy</th><th>RF Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(traccc)+'</td><td>'+color+str(teaccc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
   
    LABELS = ['Yes', 'No'] 
    conf_matrix = confusion_matrix(Y_test,tepred)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("RF Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    x1 = [pre[0],recall[0]]
    y1 = [recall[0],f1[0]]
    z1 = [f1[0],pre[0]]
    X1 = teaccc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("RF - Precision - Recall - F1 Measure")
    plt.show()
    
    
    return render_template("ViewAccuracy.html",error=output)

@app.route('/TrainCNN')
def TrainCNN():
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17,model,accmod,cnntestacc,model1,model2,model3,model5,model6
    global cnn_teacc,cnn_tracc
    # Define the input shape
   
    
    input_shape = (12,1)
    
    # Create a sequential model
    model = tf.keras.models.Sequential()
    model1 = tf.keras.models.Sequential()
    model2= tf.keras.models.Sequential()
    model3 = tf.keras.models.Sequential()
    model5= tf.keras.models.Sequential()
    model6= tf.keras.models.Sequential()
    # Add a 1D convolutional layer with 32 filters and a kernel size of 3
    model.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model1.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model2.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model3.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model5.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))
    model6.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=input_shape))

    # Add a max pooling layer with a pool size of 2
    model.add(MaxPooling1D(pool_size=2))
    model1.add(MaxPooling1D(pool_size=2))
    model2.add(MaxPooling1D(pool_size=2))
    model3.add(MaxPooling1D(pool_size=2))
    model5.add(MaxPooling1D(pool_size=2))
    model6.add(MaxPooling1D(pool_size=2))

    # Flatten the output of the convolutional layer
    model.add(Flatten())
    model1.add(Flatten())
    model2.add(Flatten())
    model3.add(Flatten())
    model5.add(Flatten())
    model6.add(Flatten())
    # Add two dense layers with 64 and 32 nodes, respectively
    model.add(Dense(64, activation="relu"))
    model1.add(Dense(64, activation="relu"))
    model2.add(Dense(64, activation="relu"))
    model3.add(Dense(64, activation="relu"))
    model5.add(Dense(64, activation="relu"))
    model6.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model1.add(Dense(32, activation="relu"))
    model2.add(Dense(32, activation="relu"))
    model3.add(Dense(32, activation="relu"))
    model5.add(Dense(32, activation="relu"))
    model6.add(Dense(32, activation="relu"))


    # Add a final dense layer with 1 node and a linear activation function for regression
    model.add(Dense(1, activation="linear"))
    model1.add(Dense(1, activation="linear"))
    model2.add(Dense(1, activation="linear"))
    model3.add(Dense(1, activation="linear"))
    model5.add(Dense(1, activation="linear"))
    model6.add(Dense(1, activation="linear"))

    # Compile the model with mean squared error loss and Adam optimizer
    #model.compile(loss="mse", optimizer="adam")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model6.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the numerical dataset
    #model.fit(X_train, Y_train, batch_size=64, epochs=20, validation_data=(X_test, Y_test))
    model1.fit(X_train1, Y_train1, batch_size=64, epochs=20, validation_data=(X_test1, Y_test1))     
    model2.fit(X_train2, Y_train2, batch_size=64, epochs=20, validation_data=(X_test2, Y_test2))
    model3.fit(X_train3, Y_train3, batch_size=64, epochs=20, validation_data=(X_test3, Y_test3))
    model5.fit(X_train5, Y_train5, batch_size=64, epochs=20, validation_data=(X_test5, Y_test5))
    model6.fit(X_train6, Y_train6, batch_size=64, epochs=20, validation_data=(X_test6, Y_test6))

    # Evaluate the model on a test set
    tepred = model.predict(X_test)
    tepred1 = model1.predict(X_test1)
    tepred2 = model2.predict(X_test2)
    tepred3 = model3.predict(X_test3)
    tepred5 = model5.predict(X_test5)
    tepred6 = model6.predict(X_test6)
    #cnnaccu = accuracy_score(Y_test,scoress)
    trpred1 = model1.predict(X_train1)
    trpred2 = model2.predict(X_train2)
    trpred3 = model3.predict(X_train3)
    trpred5 = model5.predict(X_train5)
    trpred6 = model6.predict(X_train6)

    #scores = model.evaluate(X_test, Y_test1, verbose=0)
    scores1 = model1.evaluate(X_test1, Y_test1, verbose=0)
    scores2 = model2.evaluate(X_test2, Y_test2, verbose=0)
    scores3 = model3.evaluate(X_test3, Y_test3, verbose=0)
    scores5 = model5.evaluate(X_test5, Y_test5, verbose=0)
    scores6 = model6.evaluate(X_test6, Y_test6, verbose=0)
    print('Test accuracy:', scores1)
    
    loss, accuracy = model1.evaluate(X_train, Y_train1, verbose=1)
    loss1, accuracy1 = model1.evaluate(X_train1, Y_train1, verbose=1)
    loss2, accuracy2 = model2.evaluate(X_train2, Y_train2, verbose=1)
    loss3, accuracy3 = model3.evaluate(X_train3, Y_train3, verbose=1)
    loss5, accuracy5 = model5.evaluate(X_train5, Y_train5, verbose=1)
    loss6, accuracy6 = model6.evaluate(X_train6, Y_train6, verbose=1)  
    cnntrainacclist=[accuracy1,accuracy2,accuracy3,accuracy5,accuracy6]
    cnntrainacc = (sum(cnntrainacclist)/len(cnntrainacclist) )*100   
    
    
    loss_v, cnnaccuracy_v = model1.evaluate(X_test, Y_test1, verbose=1)
    loss_v1, cnnaccuracy_v1 = model1.evaluate(X_test1, Y_test1, verbose=1)
    loss_v2, cnnaccuracy_v2 = model2.evaluate(X_test2, Y_test2, verbose=1)
    loss_v3, cnnaccuracy_v3 = model3.evaluate(X_test3, Y_test3, verbose=1)
    loss_v5, cnnaccuracy_v5 = model5.evaluate(X_test5, Y_test5, verbose=1)
    loss_v6, cnnaccuracy_v6 = model6.evaluate(X_test6, Y_test6, verbose=1)
    cnntestacclist=[cnnaccuracy_v1,cnnaccuracy_v2,cnnaccuracy_v3,cnnaccuracy_v5,cnnaccuracy_v6]
    cnntestacc = (sum(cnntestacclist)/len(cnntestacclist) )*100
    
    
    
    print("train:accuracy = %f  " % (cnntrainacc))
    print("Test: accuracy = %f  " % (cnntestacc))
    accmod["CNN"] = cnntestacc
    
    cml1 = []
    for i1 in tepred1:cml1.append(round(i1[0]))
        
        
    cml2 = []
    for i2 in tepred2:cml2.append(round(i2[0]))
        
    cml3= []
    for i3 in tepred3:cml3.append(round(i3[0]))
        
    cml5 = []
    for i5 in tepred5:cml5.append(round(i5[0]))
        
    cml6 = []
    for i6 in tepred6:cml6.append(round(i6[0]))
        
    cm1 = confusion_matrix(cml1,Y_test1)
    cm2 = confusion_matrix(cml2,Y_test2)
    cm3 = confusion_matrix(cml3,Y_test3)
    cm5 = confusion_matrix(cml5,Y_test5)
    cm6 = confusion_matrix(cml6,Y_test6)
    
    cmll = [cml1,cml2,cml3,cml5,cml6]
    rr = r.randint(0,4)
    Y_testl = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]
    
    
    XLABELS = ['Yes', 'No']
    YLABELS = ['No','Yes']
    conf_matrix = confusion_matrix(Y_test,cmll[rr])
    #conf_matrix = cmll[rr]
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = XLABELS, yticklabels = YLABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("CNN Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()
    plt.close()
    
    pre = conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]) #precision_score(cmll[rr],Y_testl[rr])
    recall = conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][1])#recall_score(cmll[rr],Y_testl[rr])
    f1 = 2*(pre*recall)/(pre+recall)#f1_score(cmll[rr],Y_testl[rr])
    
    print(pre,recall,f1,"\n",conf_matrix)
    
    x1 = [pre,recall]
    y1 = [recall,f1]
    z1 = [f1,pre]
    X1 = cnntestacc

    plt.plot(x1, y1, label = "Precision - Recall")
    plt.plot(y1, z1, label = "Recall - F1 Measure")
    plt.plot(x1, z1, label = "F1 Measure - Precision")
    plt.legend()
    plt.title("CNN - Precision - Recall - F1 Measure")
    plt.show()
    
    cnn_teacc=cnntestacc  
    cnn_tracc=cnntrainacc    
    
    
    cnnaccuracy_vv = cnntestacc
    
    output = cnntestacc
    color = '<font size="" color="black">'
    output = '<center><h2>CNN</h2><br><br><table border="1" align="center">'
    output+='<tr><th>CNN Training Accuracy</th><th>CNN Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
    output+='<tr><td>'+color+str(cnntrainacc)+'</td><td>'+color+str(cnntestacc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
    output+='</table><br><a href="TrainModel">Back</a></center>'
    
    
    
    return render_template("ViewAccuracy.html",error=output)


@app.route('/TrainMCNNN',methods=["GET","POST"])

def TrainMCNNN():
    global classifier,MLP,RF,LGBM,KNN,GBC,ETC,mcnntestacc,mcnn_teacc,mcnn_tracc
    global le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, le13, le14, le15, le16, le17,mmodel,accmod,mmodel1,mmodel2,mmodel3,mmodel5,mmodel6
    if request.method == 'POST':
        ks = int(request.form['ks'])
        activ = request.form['activ']
        optim = request.form['optimm']
        bs = int(request.form['bs'])
        
        
        eps = int(request.form['eps'])
    
        
    
    # Define the input shape
        input_shape = (12, 1)

    # Create a sequential model
        mmodel = tf.keras.models.Sequential()
        mmodel1 = tf.keras.models.Sequential()
        mmodel2 = tf.keras.models.Sequential()
        mmodel3 = tf.keras.models.Sequential()
        mmodel5 = tf.keras.models.Sequential()
        mmodel6 = tf.keras.models.Sequential()    
    
    
    # Add a 1D convolutional layer with 32 filters and a kernel size of 3
        mmodel.add(Conv1D(64, kernel_size=ks, activation="relu", input_shape=input_shape))
        mmodel1.add(Conv1D(64, kernel_size=ks, activation="relu", input_shape=input_shape))
        mmodel2.add(Conv1D(64, kernel_size=ks, activation="relu", input_shape=input_shape))
        mmodel3.add(Conv1D(64, kernel_size=ks, activation="relu", input_shape=input_shape))
        mmodel5.add(Conv1D(64, kernel_size=ks, activation="relu", input_shape=input_shape))
        mmodel6.add(Conv1D(64, kernel_size=ks, activation="relu", input_shape=input_shape))      
        
        
        

    # Add a max pooling layer with a pool size of 2
        mmodel.add(MaxPooling1D(pool_size=2))
        mmodel1.add(MaxPooling1D(pool_size=2))
        mmodel2.add(MaxPooling1D(pool_size=2))
        mmodel3.add(MaxPooling1D(pool_size=2))
        mmodel5.add(MaxPooling1D(pool_size=2))
        mmodel6.add(MaxPooling1D(pool_size=2))       
        
        
        

    # Flatten the output of the convolutional layer
        mmodel.add(Flatten())
        mmodel1.add(Flatten())
        mmodel2.add(Flatten())
        mmodel3.add(Flatten())
        mmodel5.add(Flatten())
        mmodel6.add(Flatten())

    # Add two dense layers with 64 and 32 nodes, respectively
        mmodel.add(Dense(64, activation="relu"))
        mmodel1.add(Dense(64, activation="relu"))
        mmodel2.add(Dense(64, activation="relu"))
        mmodel3.add(Dense(64, activation="relu"))
        mmodel5.add(Dense(64, activation="relu"))
        mmodel6.add(Dense(64, activation="relu"))        
        
        
        mmodel.add(Dense(32, activation=activ))
        mmodel1.add(Dense(32, activation=activ))
        mmodel2.add(Dense(32, activation=activ))
        mmodel3.add(Dense(32, activation=activ))
        mmodel5.add(Dense(32, activation=activ))
        mmodel6.add(Dense(32, activation=activ))
        

    # Add a final dense layer with 1 node and a linear activation function for regression
        mmodel.add(Dense(1, activation="linear"))
        mmodel1.add(Dense(1, activation="linear"))
        mmodel2.add(Dense(1, activation="linear"))
        mmodel3.add(Dense(1, activation="linear"))
        mmodel5.add(Dense(1, activation="linear"))
        mmodel6.add(Dense(1, activation="linear"))

    # Compile the model with mean squared error loss and Adam optimizer
    #model.compile(loss="mse", optimizer="adam")
        mmodel.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        mmodel1.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        mmodel2.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        mmodel3.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        mmodel5.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        mmodel6.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the numerical dataset
        mmodel.fit(X_train, Y_train, batch_size=bs, epochs=eps, validation_data=(X_test, Y_test))
        mmodel1.fit(X_train1, Y_train1, batch_size=bs, epochs=eps, validation_data=(X_test1, Y_test1))
        mmodel2.fit(X_train2, Y_train2, batch_size=bs, epochs=eps, validation_data=(X_test2, Y_test2))
        mmodel3.fit(X_train3, Y_train3, batch_size=bs, epochs=eps, validation_data=(X_test3, Y_test3))
        mmodel5.fit(X_train5, Y_train5, batch_size=bs, epochs=eps, validation_data=(X_test5, Y_test5))
        mmodel6.fit(X_train6, Y_train6, batch_size=bs, epochs=eps, validation_data=(X_test6, Y_test6))

    # Evaluate the model on a test set
        scoress = mmodel.predict(X_test)
        scoress1 = mmodel1.predict(X_test1)
        scoress2 = mmodel2.predict(X_test2)
        scoress3 = mmodel3.predict(X_test3)
        scoress5 = mmodel5.predict(X_test5)
        scoress6 = mmodel6.predict(X_test6)
               
        
        
    #cnnaccu = accuracy_score(Y_test,scoress)


        #scores = mmodel.evaluate(X_test, Y_test, verbose=0)
        #print('Test accuracy:', scores)
    
        #loss, accuracy2 = mmodel.evaluate(X_test, Y_test, verbose=1)
        #loss_v, cnnaccuracy_v2 = mmodel.evaluate(X_test, Y_test, verbose=1)
        #print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy2, loss_v))
        #print("Test: accuracy = %f  ;  loss = %f" % (cnnaccuracy_v2*100, loss*100))
        #accmod["MCNN"] = cnnaccuracy_v2*100
    
    
        #cnnaccuracy_v2v = cnnaccuracy_v2*100
    
        #output = cnnaccuracy_v2v
        #color = '<font size="" color="black">'
        #output = '<center><h2>CNN</h2><br><br><table border="1" align="center">'
        #output+='<tr><th>CNN Training Accuracy</th><th>CNN Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
        #output+='<tr><td>'+color+str(cnnaccuracy_v2v)+'</td><td>'+color+str(cnnaccuracy_v2v)+'</td><td>'+color+str(0)+'</td><td>'+color+str(0)+'</td><td>'+color+str(0)+'</td></tr>'
        #output+='</table><br><a href="TrainModel">Back</a></center>'
    
        tepred = mmodel.predict(X_test)
        tepred1 = mmodel1.predict(X_test1)
        tepred2 = mmodel2.predict(X_test2)
        tepred3 = mmodel3.predict(X_test3)
        tepred5 = mmodel5.predict(X_test5)
        tepred6 = mmodel6.predict(X_test6)
    #cnnaccu = accuracy_score(Y_test,scoress)
        trpred1 = mmodel1.predict(X_train1)
        trpred2 = mmodel2.predict(X_train2)
        trpred3 = mmodel3.predict(X_train3)
        trpred5 = mmodel5.predict(X_train5)
        trpred6 = mmodel6.predict(X_train6)

    #scores = model.evaluate(X_test, Y_test1, verbose=0)
        scores1 = mmodel1.evaluate(X_test1, Y_test1, verbose=0)
        scores2 = mmodel2.evaluate(X_test2, Y_test2, verbose=0)
        scores3 = mmodel3.evaluate(X_test3, Y_test3, verbose=0)
        scores5 = mmodel5.evaluate(X_test5, Y_test5, verbose=0)
        scores6 = mmodel6.evaluate(X_test6, Y_test6, verbose=0)
        print('Test accuracy:', scores1)
    
        loss, accuracy = mmodel1.evaluate(X_train, Y_train1, verbose=1)
        loss1, accuracy1 = mmodel1.evaluate(X_train1, Y_train1, verbose=1)
        loss2, accuracy2 = mmodel2.evaluate(X_train2, Y_train2, verbose=1)
        loss3, accuracy3 = mmodel3.evaluate(X_train3, Y_train3, verbose=1)
        loss5, accuracy5 = mmodel5.evaluate(X_train5, Y_train5, verbose=1)
        loss6, accuracy6 = mmodel6.evaluate(X_train6, Y_train6, verbose=1)  
        cnntrainacclist=[accuracy1,accuracy2,accuracy3,accuracy5,accuracy6]
        mcnntrainacc = (sum(cnntrainacclist)/len(cnntrainacclist) )*100   
    
    
        loss_v, cnnaccuracy_v = mmodel1.evaluate(X_test, Y_test1, verbose=1)
        loss_v1, cnnaccuracy_v1 = mmodel1.evaluate(X_test1, Y_test1, verbose=1)
        loss_v2, cnnaccuracy_v2 = mmodel2.evaluate(X_test2, Y_test2, verbose=1)
        loss_v3, cnnaccuracy_v3 = mmodel3.evaluate(X_test3, Y_test3, verbose=1)
        loss_v5, cnnaccuracy_v5 = mmodel5.evaluate(X_test5, Y_test5, verbose=1)
        loss_v6, cnnaccuracy_v6 = mmodel6.evaluate(X_test6, Y_test6, verbose=1)
        cnntestacclist=[cnnaccuracy_v1,cnnaccuracy_v2,cnnaccuracy_v3,cnnaccuracy_v5,cnnaccuracy_v6]
        mcnntestacc = (sum(cnntestacclist)/len(cnntestacclist) )*100
    
    
    
        print("train:accuracy = %f  " % (cnntrainacc))
        print("Test: accuracy = %f  " % (cnntestacc))
        accmod["CNN"] = mcnntestacc
        
        
        cml1 = []
        for i1 in tepred1:cml1.append(round(i1[0]))
        
        
        cml2 = []
        for i2 in tepred2:cml2.append(round(i2[0]))
        
        cml3= []
        for i3 in tepred3:cml3.append(round(i3[0]))
        
        cml5 = []
        for i5 in tepred5:cml5.append(round(i5[0]))
        
        cml6 = []
        for i6 in tepred6:cml6.append(round(i6[0]))
        
        cm1 = confusion_matrix(cml1,Y_test1)
        cm2 = confusion_matrix(cml2,Y_test2)
        cm3 = confusion_matrix(cml3,Y_test3)
        cm5 = confusion_matrix(cml5,Y_test5)
        cm6 = confusion_matrix(cml6,Y_test6)
    
        cmll = [cml1,cml2,cml3,cml5,cml6]
        rr = r.randint(0,4)
        Y_testl = [Y_test1,Y_test2,Y_test3,Y_test5,Y_test6]
    
    
        LABELS = ['Yes', 'No'] 
        conf_matrix = confusion_matrix(Y_test,cmll[rr])
        #conf_matrix = cmll[rr]
        plt.figure(figsize =(6, 6)) 
        ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,2])
        plt.title("MCNN Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
        plt.close()
        
        
        pre = conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[0][1]) #precision_score(cmll[rr],Y_testl[rr])
        recall = conf_matrix[0][0]/(conf_matrix[0][0]+conf_matrix[1][1])#recall_score(cmll[rr],Y_testl[rr])
        f1 = 2*(pre*recall)/(pre+recall)#f1_score(cmll[rr],Y_testl[rr])
    
        print(pre,recall,f1,"\n",conf_matrix)
    
        x1 = [pre,recall]
        y1 = [recall,f1]
        z1 = [f1,pre]
        X1 = mcnntestacc

        plt.plot(x1, y1, label = "Precision - Recall")
        plt.plot(y1, z1, label = "Recall - F1 Measure")
        plt.plot(x1, z1, label = "F1 Measure - Precision")
        plt.legend()
        plt.title("MCNN - Precision - Recall - F1 Measure")
        plt.show()
        mcnn_teacc=mcnntestacc
        mcnn_tracc=mcnntrainacc
    
        cnnaccuracy_vv = mcnntestacc
    
        output = mcnntestacc
        color = '<font size="" color="black">'
        output = '<center><h2>MCNN</h2><br><br><table border="1" align="center">'
        output+='<tr><th>MCNN Training Accuracy</th><th>MCNN Testing Accuracy</th><th>Precision</th><th>Recall</th><th>F Measure</th></tr>'
        output+='<tr><td>'+color+str(mcnntrainacc)+'</td><td>'+color+str(mcnntestacc)+'</td><td>'+color+str(pre)+'</td><td>'+color+str(recall)+'</td><td>'+color+str(f1)+'</td></tr>'
        output+='</table><br><a href="TrainModel">Back</a></center>'
    #TrainMCNN("adadelta","softmax",5,64,25)
        return render_template("ViewAccuracy.html",error=output)
    



@app.route('/PredictAction', methods =['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        #global classifier,MLP,RF,LGBM,KNN,GBC,ETC,classifier1,classifier2,classifier3,classifier5,classifier6,MLP1,MLP2,MLP3,MLP5,MLP6,RF1,RF2,RF3,RF5,RF6,GBC1,GBC2,GBC3,GBC5,GBC5,GBC6,ETC1,ETC2,ETC3,ETC5,ETC6,LGBM1,LGBM2,LGBM3,LGBM5,LGBM6,KNN1,KNN2,KNN3,KNN5,KNN6
        global le1, le2, le3, le4, le5, le6, le7, le8
        Sex = str(request.form['t1']).strip()
        Age = request.form['t2']
        BMI = request.form['t3']
        Smoking = request.form['t4']
        AlcoholDrinking = request.form['t5']
        Fruits = request.form['t6']
        cholesterol = request.form['t7']
        #PhysicalHealth = request.form['t8']
        #MentalHealth = request.form['t9']
        Stress = request.form['t8']
        Race = request.form['t9']
        Exercise = request.form['t10']
        GenHealth = request.form['t11']
        SleepTime = request.form['t12']
        
        Heart = Sex
     
        
        modelss = request.form['models']
        data = 'Sex,Age,BMI,Smoking,AlcoholDrinking,Fruits,cholesterol,Stress,Race,Exercise,GenHealth,SleepTime,HeartDisease\n'
        data+=Sex+","+Age+","+BMI+","+Smoking+","+AlcoholDrinking+","+Fruits+","+cholesterol+","+Stress+","+Race+","+Exercise+","+GenHealth+","+SleepTime+","+Heart
        
        f = open("test.csv", "w")
        f.write(data)
        f.close()
        test = pd.read_csv("test.csv")
		
		
        #columnss = ['Eating_Habits','Physical_Activity','BMI','Stress','Sleep','Smoking','Alcohol','Gender','Age']
        df = pd.read_csv(dapath)
        columnss = df.columns[0:12]
        dff = test
        
        #norm7 = normalize([df[df.columns[7]]])
        norm1 = normalize([dff[dff.columns[1]]])
        norm2 = normalize([dff[dff.columns[2]]])
        #norm8 = normalize([df[df.columns[8]]])
        norm11 = normalize([dff[dff.columns[11]]])
        #norm5 = normalize([df[df.columns[5]]])
    
        #df[df.columns[7]]=norm7[0][0:]
        dff[dff.columns[1]]=norm1[0][0:]
        dff[dff.columns[2]]=norm2[0][0:]
        dff[dff.columns[11]]=norm11[0][0:]
        #df[df.columns[8]]=norm8[0][0:]
        #df[df.columns[5]]=norm5[0][0:]

        le = LabelEncoder()

        columnss = df.columns
        dff[columnss[0]]= le.fit_transform(dff[columnss[0]])

        dff[columnss[3]]= le.fit_transform(dff[columnss[3]])

        dff[columnss[4]]= le.fit_transform(dff[columnss[4]])

        dff[columnss[9]]= le.fit_transform(dff[columnss[9]])
        dff[columnss[7]]= le.fit_transform(dff[columnss[7]])
        dff[columnss[8]]= le.fit_transform(dff[columnss[8]])

        dff[columnss[10]]= le.fit_transform(dff[columnss[10]])

        #df[columnss[11]]= le.fit_transform(df[columnss[11]])

        #df[columnss[12]]= le.fit_transform(df[columnss[12]])
        #df[columnss[13]]= le.fit_transform(df[columnss[13]])

        #df[columnss[19]]= le.fit_transform(df[columnss[17]])
        #df[columnss[14]]= le.fit_transform(df[columnss[14]])
        #df[columnss[15]]= le.fit_transform(df[columnss[15]])
        #df[columnss[16]]= le.fit_transform(df[columnss[16]])

        #dff[columnss[17]]= le.fit_transform(dff[columnss[17]])
        #df[columnss[18]]= le.fit_transform(df[columnss[18]])

        print(dff)
        
        
        classifierr = []
        MLPP =[]
        RFF =[]
        modelll=[]
        mmodelll =[]
        ETCC =[]
        GBCC =[]
        LGBMM=[]
        KNNN =[]
        
        
        if modelss == "classifier":classifierr = [classifier1,classifier2,classifier3,classifier5,classifier6]
        if modelss == "MLP":MLPP =[MLP1,MLP2,MLP3,MLP5,MLP6]
        if modelss == "RF":RFF =[RF1,RF2,RF3,RF5,RF6]
        if modelss == "CNN":modelll=[model1,model2,model3,model5,model6]
        if modelss == "MCNN":mmodelll =[mmodel1,mmodel2,mmodel3,mmodel5,mmodel6]
        if modelss == "ETC":ETCC =[ETC1,ECT2,ETC3,ETC5,ETC6]
        if modelss == "GBC":GBCC =[GBC1,GBC2,GBC3,GBC5,GBC6]
        if modelss == "LGBM":LGBMM=[LGBM1,LGBM2,LGBM3,LGBM5,LGBM6]
        if modelss == "KNN":KNNN =[KNN1,KNN2,KNN3,KNN5,KNN6]
        models = 0
        
        
            
            
        #test = dff[dff.columns[0:12]]
        #test = test.values
        #test = normalize(test)
		#print(test)
		
		
	    
            
        test = dff[dff.columns[0:12]]
        test = test.values
		
		
        if modelss == "classifier":models=classifierr
        elif modelss == "MLP":models=MLPP
        elif modelss == "RF":models=RFF
        elif modelss == "CNN":models=modelll
        elif modelss == "MCNN":models=mmodelll
        elif modelss == "KNN":models=KNNN
        elif modelss == "GBC":models=GBCC   
        elif modelss == "ETC":models=ETCC
        elif modelss == "LGBM":models=LGBMM
            
        predicti1 = ""
        predicti2 = ""
        predicti3 = ""
        predicti5 = ""
        predicti6 = ""
        
        #predicti = models.predict(test)
        
        
        predicti1 = models[0].predict(test)
        predicti2 = models[1].predict(test)
        predicti3 = models[2].predict(test)
        predicti5 = models[3].predict(test)
        predicti6 = models[4].predict(test)
		
        #print(str(predicti))
        msg1 = 0
        msg2 = 0
        msg3 = 0
        msg5 = 0
        msg6 =0
        
		
        #if round(predicti[0]) == 0:msg1 = 'No'

        #if round(predicti[0]) == 1:msg1 = 'Yes'
        
        Heart = msg1
        if modelss=="CNN" or modelss=="MCNN":
          if round(predicti1[0][0]) == 0:msg1 = 'No'
          if round(predicti1[0][0]) == 1:msg1 = 'Yes'
            
          if round(predicti2[0][0]) == 0:msg2 = 'No'
          if round(predicti2[0][0]) == 1:msg2 = 'Yes'
        
        
          if round(predicti3[0][0]) == 0:msg3 = 'No'
          if round(predicti3[0][0]) == 1:msg3 = 'Yes'
            
          if round(predicti5[0][0]) == 0:msg5 = 'No'
          if round(predicti5[0][0]) == 1:msg5 = 'Yes'
            
          if round(predicti6[0][0]) == 0:msg6 = 'No'
          if round(predicti6[0][0]) == 1:msg6 = 'Yes'
        else:
          if round(predicti1[0]) == 0:msg1 = 'No'
          if round(predicti1[0]) == 1:msg1 = 'Yes'
            
          if round(predicti2[0]) == 0:msg2 = 'No'
          if round(predicti2[0]) == 1:msg2 = 'Yes'
        
        
          if round(predicti3[0]) == 0:msg3 = 'No'
          if round(predicti3[0]) == 1:msg3 = 'Yes'
            
          if round(predicti5[0]) == 0:msg5 = 'No'
          if round(predicti5[0]) == 1:msg5 = 'Yes'
            
          if round(predicti6[0]) == 0:msg6 = 'No'
          if round(predicti6[0]) == 1:msg6 = 'Yes'        
        
        
        
        
        
        
        
        
        
        
       
        
        


        f = open("test.csv", "w")
        f.write(data)
        f.close()
        test = pd.read_csv("test.csv")
			
		
        color = '<font size="" color="black">'
        output = '<center><h2>Predicted Diseases</h2><br><br><table border="1" align="center" style"width:200px;height:50px;">'
        output+='<tr><th>&nbsp;&nbsp;&nbsp;&nbsp;Asthma&nbsp;&nbsp;&nbsp;&nbsp;</th><th>&nbsp;&nbsp;&nbsp;&nbsp;KidneyDisease&nbsp;&nbsp;&nbsp;&nbsp;</th><th>&nbsp;&nbsp;&nbsp;&nbsp;SkinCancer&nbsp;&nbsp;&nbsp;&nbsp;</th><th>&nbsp;&nbsp;&nbsp;&nbsp;Stroke&nbsp;&nbsp;&nbsp;&nbsp;</th><th>&nbsp;&nbsp;&nbsp;&nbsp;HeartDisease&nbsp;&nbsp;&nbsp;&nbsp;</th></tr>'
        output+='<tr><td><center>'+color+str(msg1)+'</center></td><td><center>'+color+str(msg2)+'</center></td><td><center>'+color+str(msg3)+'</center></td><td><center>'+color+str(msg5)+'</center></td><td><center>'+color+str(msg6)+'</center></td></tr></table><br><br><br></center>'
        output+='<center><a href="TrainModel">Back</a><br><br><br></center>'
        return render_template("ViewAccuracy.html",error=output)	

				
		
@app.route("/AccuracyGraph")
def AccuracyGraph():
    
    X = [svm_teacc,mlp_teacc,knn_teacc,gb_teacc,etc_teacc,rf_teacc,lgbm_teacc,cnn_teacc,mcnn_teacc]
    Q = [svm_tracc,mlp_tracc,knn_tracc,gb_tracc,etc_tracc,rf_tracc,lgbm_tracc,cnn_tracc,mcnn_tracc]
    plt.bar(X[0],Q[0] ,label = "SVM")
    plt.bar(X[1],Q[1], label = "MLP")
    plt.bar(X[2],Q[2], label = "KNN")
    plt.bar(X[3], Q[3],label = "GBC")
    plt.bar(X[4], Q[4],label = "ETC")
    plt.bar(X[5], Q[5],label = "RF")
    plt.bar(X[6], Q[6],label = "LGBM")
    plt.bar(X[7], Q[7],label = "CNN")
    plt.bar(X[8], Q[8],label = "MCNNs")
    plt.xlabel("TEST ACCURACY")
    plt.ylabel("TRAIN ACCURACY")
    plt.legend()
    plt.title("AccuracyGraph")
    plt.show()
    
    
    
    accmod = {"svm":svm_teacc,"mlp":mlp_teacc,"rf":rf_teacc,"KNN":knn_teacc,"GB":gb_teacc,"LGBM":lgbm_teacc,"ETC":etc_teacc,"CNN":cnnaccuracy_vv,"MCNN":cnnaccuracy_v2v}
    accmodd = list(accmod.values())
    accvalue = max(accmod.values())
    accmodel = list(accmod.keys())[list(accmod.values()).index(accvalue)]

    print("maximum accuracy is :",accvalue,"\nmodel name is :",accmodel)
    color = '<font size="" color="black">'
    #output = '<center><br>SVM Accuracy : '+color+str(accmodd[0])+'<br>MLP Accuracy : '+color+str(accmodd[1])+'<br>RF Accuracy : '+color+str(accmodd[2])+'<br>'
    output+= '<br>Best Model is : '+color+str(accmodel)+'  &  Accuracy is : '+color+str(accvalue)+'</center><br>'
    return render_template("ViewAccuracy.html",error=output)
	
@app.route('/TrainMCNN')

def TrainMCNN():
    return render_template("Mcnn.html")


@app.route("/Predict")
def Predict():
    
    print(X_test6)
    return render_template("Predict.html")

@app.route("/index")
def index():
	return render_template("index.html")

@app.route("/Login")
def Login():
	return render_template("Login.html")
	
	
@app.route("/Register")
def Register():
	return render_template("Register.html")

@app.route('/UserLogin', methods =['GET', 'POST'])
def UserLogin():
    
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        
    
        e = open("db.txt","r")
        re = e.readlines()

        e.close()


        te = "".join(re)
        
        if username in te and password in te:return render_template("AdminScreen.html",error='Welcome '+username)
            
		#if username == 'COMG09' and password == 'COMG09':
		   #return render_template("AdminScreen.html",error='Welcome '+username)
        else:return render_template("Login.html",error='Invalid Login Details')

@app.route('/UserRegister', methods =['GET', 'POST'])
def UserRegister():
    
    if request.method == 'POST':
        username = request.form['t1']
        password = request.form['t2']
        
        fff = open("db.txt","a")
        ff = open("db.txt","r")
        ve = ff.readlines()
        tve = "".join(ve)
        if username in tve and password in tve:return render_template("Register.html",error='User Already Exist')
                
        else:
            fff.write(username)
            fff.write(password)
            print("Register Success")
            fff.close()
            ff.close()
            return render_template("AdminScreen.html",error='Welcome '+username)
	#else:return render_template("Register.html",error='Invalid Register')

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=8080, debug=True)
