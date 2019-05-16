from genericModele import GenericModele
from model_all_pipeline import Model_all_pipeline
import pyriemann
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv
import os
import dataLoader
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv1D,Dropout,MaxPooling1D,GlobalAveragePooling1D
from keras import backend as K
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

#entrainement sur 1 et évaluation sur 1(le meme)
def test_all_pipeline_1_1(mesDonnees):
    print("debut test all pipeline")
    clf = LogisticRegression(penalty="l1",max_iter=10000,C=0.05)
    m = Model_all_pipeline(clf,mesDonnees)
    new_labels = m.concat_labels(mesDonnees.labels)
    score = cross_validate(m, mesDonnees.data[0], new_labels, cv=5,scoring="f1")
    print(score)
    """
    {'fit_time': array([4.58559537, 4.3623178 , 4.62051487, 4.55850363, 4.36017895]), 'score_time': array([0.15163422, 0.16563201, 0.17172813, 0.17162347, 0.17052531]), 'test_score': array([0.79389313, 0.76033058, 0.72072072, 0.67889908, 0.70175439])}
    """

#Entrainement sur 1 évaluation sur 15
def test_all_pipeline_1_15(myFileList):
    for indice_fichier in range(len(myFileList)):
        fichier_train = [myFileList[indice_fichier]]
        fichier_test = myFileList[:indice_fichier]

        if(indice_fichier != len(myFileList)-1 ):
            fichier_test+=myFileList[indice_fichier+1:]

        mesDonnees_train = dataLoader.DataLoader(fichier_train,concatenate=False)
        mesDonnees_test = dataLoader.DataLoader(fichier_test,concatenate=True)
        mesDonnees_train.filtre_Matrice()
        mesDonnees_test.filtre_Matrice()

        clf = LogisticRegression(penalty="l1",max_iter=10000,C=0.05)
        m = Model_all_pipeline(clf,mesDonnees_train)
        new_labels_train = m.concat_labels(mesDonnees_train.labels)
        new_labels_test = m.concat_labels(mesDonnees_test.labels)
        m.fit(mesDonnees_train.data[0],new_labels_train)
        pred = m.predict(mesDonnees_test.data)
        score = classification_report(new_labels_test, pred, output_dict=True)
        file = open(str(indice_fichier)+".txt","w")
        file.write(str(score))
        file.close()
        print(score)

#Entrainement sur 15 évaluation sur 1
def test_all_pipeline_15_1(myFileList):
    for indice_fichier in range(len(myFileList)):
        fichier_test = [myFileList[indice_fichier]]
        fichier_train = myFileList[:indice_fichier]

        if(indice_fichier != len(myFileList)-1 ):
            fichier_train+=myFileList[indice_fichier+1:]

        mesDonnees_train = dataLoader.DataLoader(fichier_train,concatenate=False)
        mesDonnees_test = dataLoader.DataLoader(fichier_test,concatenate=False)
        print("----------------------------")
        print(np.array(mesDonnees_train.data).shape)
        print(np.array(mesDonnees_train.labels).shape)
        print(np.array(mesDonnees_test.data).shape)
        print(np.array(mesDonnees_test.labels).shape)
        print("----------------------------")
        mesDonnees_train.filtre_Matrice()
        mesDonnees_test.filtre_Matrice()
        print("----------------------------")
        print(np.array(mesDonnees_train.data).shape)
        print(np.array(mesDonnees_train.labels).shape)
        print(np.array(mesDonnees_test.data).shape)
        print(np.array(mesDonnees_test.labels).shape)
        print("----------------------------")

        clf = LogisticRegression(penalty="l1",max_iter=10000,C=0.05)
        m = Model_all_pipeline(clf,mesDonnees_train)
        new_labels_train = m.concat_labels(mesDonnees_train.labels)
        new_labels_test = m.concat_labels(mesDonnees_test.labels)
        m.fit(mesDonnees_train.data,new_labels_train)
        pred = m.predict(mesDonnees_test.data)
        score = classification_report(new_labels_test, pred, output_dict=True)
        file = open(str(indice_fichier)+".txt","w")
        file.write(str(score))
        file.close()
        print(score)

#Entrainement sur une partie des 16 évaluation sur une partie des 16
def test_all_pipeline_16_16(myFileList):
    #minimum 580 trial par sujet donc on coupe pour que ça soit plus simple( 5 partie de 116)

    for mult in range(5):
        indice = int(mult*116)
        mesDonnees_train = dataLoader.DataLoader(myFileList,concatenate=False)
        mesDonnees_test = dataLoader.DataLoader(myFileList,concatenate=False)

        for k in range(len(mesDonnees_train.data)):
            mesDonnees_train.data[k]=np.concatenate((mesDonnees_train.data[k][:indice],mesDonnees_train.data[k][indice+116:580]))
            mesDonnees_train.labels[k]=np.concatenate((mesDonnees_train.labels[k][:indice],mesDonnees_train.labels[k][indice+116:580]))
            mesDonnees_test.data[k]=mesDonnees_test.data[k][indice:indice+116]
            mesDonnees_test.labels[k]=mesDonnees_test.labels[k][indice:indice+116]

        mesDonnees_train.filtre_Matrice()
        mesDonnees_test.filtre_Matrice()

        clf = LogisticRegression(penalty="l1",max_iter=10000,C=0.05)
        m = Model_all_pipeline(clf,mesDonnees_train)
        new_labels_train = m.concat_labels(mesDonnees_train.labels)
        new_labels_test = m.concat_labels(mesDonnees_test.labels)
        m.fit(mesDonnees_train.data,new_labels_train)
        pred = m.predict(mesDonnees_test.data)
        score = classification_report(new_labels_test, pred, output_dict=True)
        file = open(str(mult)+".txt","w")
        file.write(str(score))
        file.close()
        print(score)

#myFileList_train=["data/train/train_subject16.mat"]
myFileList = getListOfFiles("data/train/")
test_all_pipeline_16_16(myFileList)

"""
mesDonnees= dataLoader.DataLoader(myFileList_train,concatenate=False)
mesDonnees.filtre_Matrice()

proportion_train = 0.8
len_train = int(proportion_train*len(mesDonnees_train.data[0]))
indice_shuffle = np.random.permutation(len(mesDonnees_train.data[0]))

mesDonnees_test = copy.deepcopy(mesDonnees_train)

mesDonnees_train.data[0]=mesDonnees_train.data[0][indice_shuffle]
mesDonnees_train.labels[0]=mesDonnees_train.labels[0][indice_shuffle]

mesDonnees_train.data[0]=mesDonnees_train.data[0][:len_train]
mesDonnees_train.labels[0]=mesDonnees_train.labels[0][:len_train]

mesDonnees_test.data[0]=mesDonnees_test.data[0][indice_shuffle]
mesDonnees_test.labels[0]=mesDonnees_test.labels[0][indice_shuffle]

mesDonnees_test.data[0]=mesDonnees_test.data[0][len_train:]
mesDonnees_test.labels[0]=mesDonnees_test.labels[0][len_train:]
"""

#test_all_pipeline_1_1(mesDonnees)
