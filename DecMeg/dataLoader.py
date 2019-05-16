import mne
import numpy as np
import os
import sys
import warnings
from scipy.io import loadmat
import numpy as np
from scipy.signal import butter, lfilter

class DataLoader():
    def __init__(self,fileList,test_data=False,concatenate=True):
        self.data = None
        self.labels=None
        self.list_id=None
        self.concat = concatenate
        if(concatenate):
            self.load_data_from_multiple_file(fileList,test_data)
        else:
            self.load_data_from_multiple_file_multipleDim(fileList,test_data)

    def filtre_Matrice(self):
        if(self.concat):
            self.data = np.swapaxes(self.data,1,2)
            new_data=[]
            for trial in self.data:
                tmp=[]
                for channel in trial:
                    tmp.append(self.butter_bandpass_filter(channel,1,20,250,5))
                new_data.append(tmp)
            self.data=np.array(new_data)
            self.data = np.swapaxes(self.data,1,2)
        else:
            print(len(self.data))
            for k in range(len(self.data)):
                subject = self.data[k]
                print(subject.shape)
                subject = np.swapaxes(subject,1,2)
                new_data=[]
                for trial in subject:
                    tmp=[]
                    for channel in trial:
                        tmp.append(self.butter_bandpass_filter(channel,1,20,250,5))
                    new_data.append(tmp)
                subject=np.array(new_data)
                subject = np.swapaxes(subject,1,2)
                self.data[k]=subject


    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a


    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def load_data_from_multiple_file(self,fileList,test_data):
        all_data=np.array([])
        all_labels=np.array([])
        for f in fileList:
            self.load_data_from_file(f,test_data)
            if(len(all_data)==0):
                all_data = self.data
                all_labels = self.labels
                all_list_id=self.list_id
            else:
                all_data = np.concatenate((all_data,self.data))
                if(test_data==False):
                    all_labels = np.concatenate((all_labels,self.labels))
                else:
                    all_list_id = np.concatenate((all_list_id,self.list_id))
        self.data = all_data
        self.labels = all_labels
        self.list_id= all_list_id

    def load_data_from_multiple_file_multipleDim(self,fileList,test_data):
        for f in fileList:
            ancien_data=self.data
            ancien_labels=self.labels
            ancien_list_id = self.list_id
            self.load_data_from_file(f,test_data)
            if(ancien_data is None):
                self.data = [self.data]
                self.labels = [self.labels]
                self.list_id= [self.list_id]
            else:
                ancien_data.append(self.data)
                self.data=ancien_data
                if(test_data==False):
                    ancien_labels.append(self.labels)
                    self.labels = ancien_labels
                else:
                    ancien_list_id.append(self.list_id)
                    self.list_id = ancien_list_id
            if(self.data is not None):
                print(len(self.data))



    def load_data_from_file(self,fileName,test_data):
        frequence=250
        data = loadmat(fileName)
        #(data.keys())
        #dict_keys(['__header__', '__version__', '__globals__', 'tmin', 'tmax', 'sfreq', 'y', 'X'])
        donnees = data["X"]
        if(test_data==False):
            labels = data["y"]
        # (trial x channel x time) of size 580 x 306 x 375.
        print("On test de prendre uniquement 1 seconde apres le stimulus au lieux de 1.5seconde qui commence 0.5 avant le stimulus")
        self.data = np.array([[k[125:] for k in d] for d in donnees])
        if(test_data==False):
            self.labels = labels
            self.labels=np.reshape(self.labels,(self.labels.shape[0]))
        self.data = np.swapaxes(self.data,1,2)
        if(test_data==True):
            list_id=[]
            file_num=int(''.join(list(filter(str.isdigit, fileName))))
            for i in range(len(self.data)):
                list_id.append(int(str(file_num)+str(f'{i:03}')))
            self.list_id=list_id
