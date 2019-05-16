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


def write_to_file(name,pred,list_id):
    #Partie d'écriture de la prédiction sur un fichier
    file_sortie = open("resultats/temporaire/submission_kaggle-tmp7.csv","w")
    file_sortie.write("Id,Prediction\n")
    for k in range(len(pred)):
        file_sortie.write(str(list_id[k])+","+str(pred[k])+"\n")
    file_sortie.close()


def test_all_pipeline(mesDonnees_train,mesDonnees_test):
    print("debut test all pipeline")
    clf = LogisticRegression(penalty="l1",max_iter=10000,C=0.05)
    m = Model_all_pipeline(clf,mesDonnees_train)
    m.fit_representation()
    new_data = m.predict_representation(mesDonnees_train.data)
    new_labels = m.concat_labels(mesDonnees_train.labels)
    new_list_id = m.concat_labels(mesDonnees_test.list_id)
    #s = m.f1Score(new_data,new_labels)
    #print("f1score : ",s)
    m.fit(new_data,new_labels)
    new_data_test = m.predict_representation(mesDonnees_test.data)
    pred = m.predict(new_data_test)
    write_to_file("prediction_test.csv",pred,new_list_id)

#myFileList_train=["data/train/train_subject01.mat","data/train/train_subject02.mat"]
myFileList_train = getListOfFiles("data/train/")
myFileList_test =  getListOfFiles("data/test/")


mesDonnees = dict()
mesDonnees_train = dataLoader.DataLoader(myFileList_train,concatenate=False)
mesDonnees_test = dataLoader.DataLoader(myFileList_test,concatenate=False,test_data=True)
print("j'ai enlevé le filtre passe bande car les matrices netaits plus définit positives apres")
mesDonnees_train.filtre_Matrice()
mesDonnees_test.filtre_Matrice()
print("Attention nouveau pre processing uniquement pour pyriemann")
test_all_pipeline(mesDonnees_train,mesDonnees_test)
