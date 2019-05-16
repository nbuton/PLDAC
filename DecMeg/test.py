import dataLoader
import numpy

myFileList=["data/train/train_subject01.mat"]
mesDonnees=dataLoader.DataLoader(myFileList)
mesDonnees.pre_processing()
print(mesDonnees.data.shape)
print(mesDonnees.labels.shape)
