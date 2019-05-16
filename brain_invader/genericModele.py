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

class GenericModele():
    def __init__(self,modele,dataLoader):
        self.data=dataLoader.data
        self.labels=dataLoader.labels
        self.modele=modele

    def vectorize(self,tf=False):
        if(tf):
            self.data=self.data.reshape((len(self.data),8*len(self.data[0])))
        else:
            self.data=self.data.reshape((len(self.data),16*len(self.data[0])))

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
        self.data=pyriemann.estimation.Covariances().fit_transform(self.data)


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

    def balanceData(self,data,labels):
        #Duplique la classe minoritaire pour avoir le meme nombre d'exemple dans chaque classe
        argC1 = np.argwhere(labels==0).reshape((-1))
        argC2 = np.argwhere(labels==1).reshape((-1))

        nbC1 = len(argC1)
        nbC2 =len(argC2)

        multiple = int(nbC1/nbC2)
        reste = nbC1%nbC2

        for _ in range(multiple-1):
            labels = np.concatenate((labels,labels[argC2]))
            data = np.concatenate( ( data,data[argC2] ) )

        labels = np.concatenate((labels,labels[argC2[:reste]]))
        data = np.concatenate( ( data,data[argC2[:reste]] ) )

        randIndice = np.array(list(range(len(data))))
        np.random.shuffle(randIndice)

        data = data[randIndice]
        labels = labels[randIndice]

        return data,labels

    def post_processing(self,y_proba):
        #print(y_proba)
        new_y=np.array([])
        for k in range(0,len(y_proba),12):
            #1 car c'est la classe détéction
            tabTmp = [p[1] for p in y_proba[k:k+12]]
            sortIndice  = np.argsort(tabTmp)
            i1 = sortIndice[0]
            i2 = sortIndice[1]
            outTmp = np.zeros(12)
            outTmp[i1]=1
            outTmp[i2]=1
            new_y=np.concatenate((new_y,outTmp))
        return new_y

    def nombreDeBonParGroupe(self,y, y_pred):
        nb_deux=0
        nb_un=0
        nb_zeros=0
        for k in range(0,len(y_pred),12):
            tmp = y_pred[k:k+12]
            tmp_reel = y[k:k+12]
            compteur=0
            for i in range(12):
                if(tmp[i] == 1 and tmp[i]==tmp_reel[i]):
                    compteur+=1
            if(compteur==0):
                nb_zeros+=1
            elif(compteur==1):
                nb_un+=1
            elif(compteur==2):
                nb_deux+=1
        return nb_deux,nb_un,nb_zeros



    def f1Score(self,mode="nonProba"):
        skf = StratifiedKFold(n_splits=5)

        moyenne_f1 = 0
        moyenne_f1_C1=0
        moyenne_f1_C2=0
        moyenne_un=0
        moyenne_deux=0
        moyenne_zeros=0
        for train_index, test_index in skf.split(self.data, self.labels):
            fold_x_train, fold_x_test = self.data[train_index], self.data[test_index]
            fold_y_train, fold_y_test = self.labels[train_index], self.labels[test_index]

            fold_x_train,fold_y_train = self.balanceData(fold_x_train,fold_y_train)

            if(mode=="proba"):
                fold_y_train = self.changeLabelToProba(fold_y_train)

            self.modele.fit(fold_x_train, fold_y_train)

            #y_pred = self.modele.predict(fold_x_test)
            y_pred = self.modele.predict_proba(fold_x_test)
            y_pred = self.post_processing(y_pred)

            rapport = classification_report(fold_y_test, y_pred, output_dict=True)
            #macro avg : moyenne des f1 score des deux classe
            moyenne_f1 += rapport['macro avg']["f1-score"]
            moyenne_f1_C1 += rapport['0']["f1-score"]
            moyenne_f1_C2 += rapport['1']["f1-score"]
            deux,un,zeros = self.nombreDeBonParGroupe(fold_y_test, y_pred)
            moyenne_deux += deux
            moyenne_un += un
            moyenne_zeros += zeros

        nbSplit = skf.get_n_splits(self.data, self.labels)
        moyenne_f1 = moyenne_f1 / nbSplit
        moyenne_f1_C1 = moyenne_f1_C1 / nbSplit
        moyenne_f1_C2 = moyenne_f1_C2 / nbSplit

        return [moyenne_f1,moyenne_f1_C1,moyenne_f1_C2,moyenne_deux,moyenne_un,moyenne_zeros]
