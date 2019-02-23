import mne
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import pyriemann

class GenericModele():
    def __init__(self,reactT,modele):
        self.reactionTime=reactT
        self.data=None
        self.labels=None
        self.modele=modele
    def load_data_from_file(self,fileName):
        #17eme dans data ce sont les labels
        #16 premiere les electrodes
        raw = mne.io.read_raw_edf(fileName, preload=True)
        #On supprime le labels des data brut car ils sont déjà présent dans event
        dataBrut = raw._data[:16]
        dataBrut = dataBrut.swapaxes(0,1)
        events = mne.find_events(raw, initial_event=True, consecutive=True)
        nbDataPerReactTime=int(512*self.reactionTime)
        data=[]
        labels=[]
        for e in events :
            indiceDebut = e[0]
            indiceFin =  e[0]+nbDataPerReactTime
            data.append(dataBrut[indiceDebut:indiceFin])
            if(e[2]!=33285):
                labels.append(1)
            else:
                labels.append(2)

        self.data=np.array(data)
        self.labels=np.array(labels)

    def vectorize(self,tf=False):
        if(tf):
            self.data=self.data.reshape((len(self.data),8*len(self.data[0])))
        else:
            self.data=self.data.reshape((len(self.data),16*len(self.data[0])))


    def dataToCov(self):
        self.data=np.swapaxes(self.data,1,2)
        self.data=pyriemann.estimation.Covariances().fit_transform(self.data)


    def dataToTf(self):
        print(self.data.shape)
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
        print(self.data.shape)


    def dataToMoy(self,nbPoint,slider):
        dataFuture=[]
        for d in self.data:
            dataTmp=[]
            nbDecallage = int( (len(d)-nbPoint)/slider )
            for k in range(nbDecallage):
                dataTmp.append(np.mean(d[k*slider:k*slider+nbPoint]))
            dataFuture.append(np.array(dataTmp))
        self.data = np.array(dataFuture)


    def f1Score(self):
        y_pred = cross_val_predict(self.modele,self.data,self.labels,cv=5)
        rapport = classification_report(self.labels, y_pred,output_dict=True)
        return rapport['micro avg']["f1-score"]
