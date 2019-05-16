import mne
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import pyriemann
import os
import sys
import warnings
from pyriemann.tangentspace import TangentSpace

class Model_all_pipeline():
    def __init__(self,modele,dataLoader):
        self.dataLoader = dataLoader
        print("data uniquement si 1 seul fichier")
        self.data=dataLoader.data
        print(np.array(self.data).shape)
        self.labels=dataLoader.labels
        self.model_xDawn=[]
        self.model_tangentSpace=[]
        self.modele=modele


    def predict_representation(self,data):
        new_rep=None
        for k in range(len(self.model_xDawn)):
            new_data = None
            for q in range(len(data)):
                subject = data[q]
                model_xDawn_enCours = self.model_xDawn[k]
                model_tangentSpace_enCours = self.model_tangentSpace[k]
                subject = model_xDawn_enCours.transform(subject)
                subject = model_tangentSpace_enCours.transform(subject)
                if(new_data is None):
                    new_data=subject
                else:
                    new_data=np.append(new_data,subject,axis=0)
            if(new_rep is None):
                new_rep = new_data
            else:
                new_rep = np.append(new_rep,new_data,axis=1)
        return np.array(new_rep)

    def fit_representation(self):
        print(np.array(self.data).shape)
        for k in range(len(self.data)):
            subject_data = np.array(self.data[k])
            print(subject_data.shape)
            subject_labels = self.labels[k]
            model_xDawn_enCours = pyriemann.estimation.XdawnCovariances(4,xdawn_estimator='lwf')

            subject_data = model_xDawn_enCours.fit_transform(subject_data,subject_labels)
            self.model_xDawn.append(model_xDawn_enCours)
            model_tangentSpace_enCours=TangentSpace(metric='riemann')
            model_tangentSpace_enCours.fit(subject_data,subject_labels)
            self.model_tangentSpace.append(model_tangentSpace_enCours)

    def concat_labels(self,labels):
        new_labels = None
        for lab in labels:
            if(new_labels is None):
                new_labels= lab
            else:
                new_labels = np.append(new_labels,lab)
        return new_labels

    def fit(self,data,labels):
        self.fit_representation()
        new_data = self.predict_representation(data)#[data]
        print("################")
        print(np.array(new_data).shape)
        print(np.array(labels).shape)
        self.modele.fit(new_data, labels)

    def predict(self,data):
        new_data = self.predict_representation(data)#[data]
        return  self.modele.predict(new_data)

    def accuracy(self,true_y,pred_y):
        compteur=0
        for k in range(len(true_y)):
            if(true_y[k]==pred_y[k]):
                compteur+=1
        return compteur/len(true_y)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"dataLoader": self.dataLoader, "modele": self.modele}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    #Attention self.data et labels n'ont pas le bon format
    def f1Score(self,data,labels):
        skf = StratifiedKFold(n_splits=5)

        moyenne_rapport=0
        for train_index, test_index in skf.split(data, labels):
            fold_x_train, fold_x_test = data[train_index], data[test_index]
            fold_y_train, fold_y_test = labels[train_index], labels[test_index]

            self.modele.fit(fold_x_train, fold_y_train)

            y_pred = self.modele.predict(fold_x_test)

            rapport = classification_report(fold_y_test, y_pred, output_dict=True)
            acc = self.accuracy(fold_y_test, y_pred)
            print(acc)
            print(rapport)
            if(moyenne_rapport==0):
                moyenne_rapport=rapport
            else:
                for key1 in rapport.keys() :
                    for key2 in rapport[key1].keys() :
                        moyenne_rapport[key1][key2]+=rapport[key1][key2]


        nbSplit = skf.get_n_splits(data, labels)
        for key1 in rapport.keys() :
            for key2 in rapport[key1].keys() :
                moyenne_rapport[key1][key2]/= nbSplit

        return [moyenne_rapport["0"]["f1-score"],moyenne_rapport["1"]["f1-score"]]
        #return acc
