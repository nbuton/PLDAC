import mne
import numpy as np
import os
import sys
import warnings

class DataLoader():
    def __init__(self,fileList,reactionTime):
        self.data = None
        self.labels=None
        self.reactionTime=reactionTime
        warnings.filterwarnings("ignore")
        self.load_data_from_multiple_file(fileList)

    def load_data_from_multiple_file(self,fileList):
        all_data=np.array([])
        all_labels=np.array([])
        for f in fileList:
            self.load_data_from_file(f)
            if(len(all_data)==0):
                all_data = self.data
                all_labels = self.labels
            else:
                all_data = np.concatenate((all_data,self.data))
                all_labels = np.concatenate((all_labels,self.labels))
        self.data = all_data
        self.labels = all_labels

    def load_data_from_file(self,fileName):
        #17eme dans data ce sont les labels
        #16 premiere les electrodes
        f = open(os.devnull, 'w')
        old_stdout = sys.stdout
        sys.stdout = f

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
                labels.append(0)
            else:
                labels.append(1)

        self.data=np.array(data)
        self.labels=np.array(labels)
        sys.stdout = old_stdout
        print(self.data.shape)
        print(self.labels.shape)
