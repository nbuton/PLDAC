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

class GenericModele():
    def __init__(self,modele,dataLoader):
        self.data=dataLoader.data
        self.labels=dataLoader.labels
        self.modele=modele
        self.XdawnCov=None

    def vectorize(self,tf=False):
        print(self.data.shape)
        if(tf):
            self.data=self.data.reshape((len(self.data),self.data.shape[2]*self.data.shape[1]))
        else:
            self.data=self.data.reshape((len(self.data),self.data.shape[1]*self.data.shape[2]))

    def changeLabelToProba(self,labels):
        new_lab=[]
        for l in labels:
            if(l == 0):
                new_lab.append([1.,0.])#[1.,0.]
            else:
                new_lab.append([0.,1.])#[0.,1.]
        return np.array(new_lab)

    def changeProbaToLabel(self,labels):
        new_lab=[]
        #print(labels)
        for l in labels:
            if(l[0] == 1 and l[1]==0):
                new_lab.append(0)
            else:
                new_lab.append(1)
        #print(new_lab)
        return np.array(new_lab)

    def dataToCov(self):
        self.data=np.swapaxes(self.data,1,2)
        print("avant estimation Covariances")
        print(self.data.shape)
        self.data=pyriemann.estimation.Covariances('oas').fit_transform(self.data)
        print("apres Covariances")
        print(self.data.shape)

    def dataToXdawnCov(self):
        self.data=np.swapaxes(self.data,1,2)
        print("avant estimation xDawnCovariances")
        print(self.data.shape)
        self.XdawnCov = pyriemann.estimation.XdawnCovariances(4)
        self.data=self.XdawnCov.fit_transform(self.data,self.labels)
        print("apres xDawnCovariances")
        print(self.data.shape)


    def dataToTf(self):
        def spectre(signal):
            fft_val = np.abs(np.fft.fft(signal))
            return fft_val[:int(len(signal)/2)]
        def estimate_tf(matrice):
            new_mat=[]
            for ligne in matrice:
                new_mat.append(spectre(ligne))
            return np.array(new_mat)
        data=self.data
        new_mat=[]
        for matrice in data:
            new_mat.append(estimate_tf(matrice))
        self.data=np.array(new_mat)


    def dataToMoy(self,nbPoint,slider):
        dataFuture=[]
        for d in self.data:
            dataTmp=[]
            nbDecallage = int( (len(d)-nbPoint)/slider )
            for k in range(nbDecallage):
                dataTmp.append(np.mean(d[k*slider:k*slider+nbPoint]))
            dataFuture.append(np.array(dataTmp))
        self.data = np.array(dataFuture)


    def numberPerClass(self,labels):
        argC1 = np.argwhere(labels==0).reshape((-1))
        argC2 = np.argwhere(labels==1).reshape((-1))

        nbC1 = len(argC1)
        nbC2 =len(argC2)
        print("nbC1 : "+str(nbC1))
        print("nbC2 : "+str(nbC2))

    def accuracy(self,true_y,pred_y):
        compteur=0
        for k in range(len(true_y)):
            if(true_y[k]==pred_y[k]):
                compteur+=1
        return compteur/len(true_y)

    def predict(self,donnees):
        donnees=np.swapaxes(donnees,1,2)
        donnees=self.XdawnCov.transform(donnees)
        return self.modele.predict(donnees)

    def fit(self):
        self.modele.fit(self.data, self.labels)

    def score(self,donnees,labels):
        y_pred = self.modele.predict(donnees)
        rapport = classification_report(labels, y_pred, output_dict=True)
        return rapport

    def f1Score(self,mode="nonProba"):
        skf = StratifiedKFold(n_splits=5)

        moyenne_rapport=0
        for train_index, test_index in skf.split(self.data, self.labels):
            fold_x_train, fold_x_test = self.data[train_index], self.data[test_index]
            fold_y_train, fold_y_test = self.labels[train_index], self.labels[test_index]

            if(mode=="proba"):
                fold_y_train = self.changeLabelToProba(fold_y_train)


            self.modele.fit(fold_x_train, fold_y_train)

            y_pred = self.modele.predict(fold_x_test)
            #y_pred = self.modele.predict_proba(fold_x_test)

            rapport = classification_report(fold_y_test, y_pred, output_dict=True)
            acc = self.accuracy(fold_y_test, y_pred)
            print(acc)

            if(moyenne_rapport==0):
                moyenne_rapport=rapport
            else:
                for key1 in rapport.keys() :
                    if(not(isinstance(rapport[key1],float))):
                        for key2 in rapport[key1].keys() :
                            moyenne_rapport[key1][key2]+=rapport[key1][key2]


        nbSplit = skf.get_n_splits(self.data, self.labels)
        for key1 in rapport.keys() :
            if(not(isinstance(rapport[key1],float))):
                for key2 in rapport[key1].keys() :
                    moyenne_rapport[key1][key2]/= nbSplit

        return moyenne_rapport
        #return acc
