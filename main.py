from tkinter import *

import tkinter.scrolledtext as st 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from summarizer import Summarizer


def text_summarization():
    txt=text_area1.get("1.0",'end-1c')
    model = Summarizer()
    result = model(txt,num_sentences=3)
    full = ''.join(result)
    text_area2.delete("1.0",'end-1c')
    text_area2.insert("1.0",full)
    
def Prediction():
    dt1=var1.get()
    dt1=float(dt1)
    dt2=var2.get()
    dt2=float(dt2)
    dt3=var3.get()
    dt3=float(dt3)
    dt4=var4.get()
    dt4=float(dt4)
    dt5=var5.get()
    dt5=float(dt5)
    dt6=var6.get()
    dt6=float(dt6)
    dt7=var7.get()
    dt7=float(dt7)
    dt8=var8.get()
    dt8=float(dt8)
    dt9=var9.get()
    dt9=float(dt9)
    dt10=var10.get()
    dt10=float(dt10)
    dt11=var11.get()
    dt11=float(dt11)
    dt12=var12.get()
    dt12=float(dt12)
    dt13=var13.get()
    dt13=float(dt13)
    #test dataset
    xtest=[[dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10,dt11,dt12,dt13]] 
    data = pd.read_csv("Heart_Disease_Prediction.csv")
    X = data.drop('Heart Disease', axis=1)
    y = data['Heart Disease']
    #Data processing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)


    #Algorithm comparison
    algorithms = {"SVM Classifier:":SVC(kernel='linear'),"RandomForestClassifier":RandomForestClassifier(n_estimators=100, random_state=23),"KNeighborsClassifier":KNeighborsClassifier(n_neighbors=23)}

    results = {}
    for algo in algorithms:
        clf = algorithms[algo]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("%s : %f %%" % (algo, score*100))
        results[algo] = score

    best_algo = max(results, key=results.get)
    print('\nBest Algorithm is %s with a %f %%' % (best_algo, results[best_algo]*100))

    classifier = algorithms[best_algo]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(xtest)
    print(y_pred)
    
    e02.delete(0,'end')
    e02.insert(5,str(y_pred[0]))

master = Tk()
master.title('HEART DISEASE PREDICTION TEXT SUMMARIZATION & ML')
master.geometry('1366x768')


var1=StringVar()
var2=StringVar()
var3=StringVar()
var4=StringVar()
var5=StringVar()
var6=StringVar()
var7=StringVar()
var8=StringVar()
var9=StringVar()
var10=StringVar()
var11=StringVar()
var12=StringVar()
var13=StringVar()

Label(master,text='HEART DISEASE PREDICTION TEXT SUMMARIZATION & ML',foreground="red",font=('Verdana',25)).pack(side=TOP,pady=10)

b1=Button(master,borderwidth=1, relief="flat",text="Enter the text", font="verdana 15", bg="red", fg="white")
b1.place(height=40,width=150,x=30,y=100)



text_area1 = st.ScrolledText(master, width = 50, height = 25, font = ("Times New Roman",12)) 
text_area1.place(x=30,y=150)

b2=Button(master,borderwidth=1, relief="flat",text=" SUMMARIZE  ", font="verdana 15", bg="red", fg="white",command=text_summarization)
b2.place(x=460,y=100)

text_area2 = st.ScrolledText(master, width = 51, height = 25, font = ("Times New Roman",12)) 
text_area2.place(x=460,y=150)

b3=Button(master,borderwidth=1, relief="flat",text="PREDICT", font="verdana 15", bg="red", fg="white",command=Prediction)
b3.place(height=40,width=150,x=920,y=100)

e02=Entry(master,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e02.place(height=40,width=245,x=1090,y=100)


c2 = Canvas(master,bg='gray',width=415,height=475)
c2.place(x=920,y=150)

l0=Label(master,text='INPUT DATA',foreground="white",background='gray',font =('Verdana',9,'bold'))
l0.place(x=1075,y=160)

l1=Label(master,text='AGE',foreground="white",background='gray',font =('Verdana',9,'bold'))
l1.place(x=940,y=200)
e1=Entry(master,textvariable=var1,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e1.place(height=30,width=100,x=1025,y=200)

l2=Label(master,text='SEX',foreground="white",background='gray',font =('Verdana',8,'bold'))
l2.place(x=1140,y=200)
e2=Entry(master,textvariable=var2,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e2.place(height=30,width=100,x=1225,y=200)

l3=Label(master,text='CHEST PAIN TYPE',foreground="white",background='gray',font =('Verdana',6,'bold'))
l3.place(x=940,y=250)
e3=Entry(master,textvariable=var3,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e3.place(height=30,width=100,x=1025,y=250)

l4=Label(master,text='BP',foreground="white",background='gray',font =('Verdana',8,'bold'))
l4.place(x=1140,y=250)
e4=Entry(master,textvariable=var4,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e4.place(height=30,width=100,x=1225,y=250)

l5=Label(master,text='CHOLESTROL',foreground="white",background='gray',font =('Verdana',8,'bold'))
l5.place(x=940,y=300)
e5=Entry(master,textvariable=var5,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e5.place(height=30,width=100,x=1025,y=300)

l6=Label(master,text='FBS OVER 120',foreground="white",background='gray',font =('Verdana',8,'bold'))
l6.place(x=1140,y=300)
e6=Entry(master,textvariable=var6,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e6.place(height=30,width=100,x=1225,y=300)

l7=Label(master,text='EKG RESULT',foreground="white",background='gray',font =('Verdana',8,'bold'))
l7.place(x=940,y=350)
e7=Entry(master,textvariable=var7,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e7.place(height=30,width=100,x=1025,y=350)

l8=Label(master,text='MAX HR',foreground="white",background='gray',font =('Verdana',8,'bold'))
l8.place(x=1140,y=350)
e8=Entry(master,textvariable=var8,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e8.place(height=30,width=100,x=1225,y=350)

l9=Label(master,text='EXCERCISE ANGINA',foreground="white",background='gray',font =('Verdana',6,'bold'))
l9.place(x=940,y=400)
e9=Entry(master,textvariable=var9,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e9.place(height=30,width=100,x=1025,y=400)

l10=Label(master,text='ST DEPRESSION',foreground="white",background='gray',font =('Verdana',6,'bold'))
l10.place(x=1140,y=400)
e10=Entry(master,textvariable=var10,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e10.place(height=30,width=100,x=1225,y=400)

l11=Label(master,text='SLOPE OF ST',foreground="white",background='gray',font =('Verdana',6,'bold'))
l11.place(x=940,y=450)
e11=Entry(master,textvariable=var11,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e11.place(height=30,width=100,x=1025,y=450)

l12=Label(master,text='NO. OF VESSELS',foreground="white",background='gray',font =('Verdana',6,'bold'))
l12.place(x=1140,y=450)
e12=Entry(master,textvariable=var12,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e12.place(height=30,width=100,x=1225,y=450)

l13=Label(master,text='THALLIUM',foreground="white",background='gray',font =('Verdana',8,'bold'))
l13.place(x=940,y=500)
e13=Entry(master,textvariable=var13,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e13.place(height=30,width=100,x=1025,y=500)


mainloop()
 
