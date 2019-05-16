from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from genericModele import GenericModele
import dataLoader
import pyriemann
import os

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


#Partie apprentissage
myFileList=getListOfFiles("data/train")[:6]

mesDonnees=dataLoader.DataLoader(myFileList)
print("j'ai chargé les données")
mesDonnees.filtre_Matrice()
print("j'ai filtrer le signal")

m = GenericModele(pyriemann.classification.MDM(),mesDonnees)
m.dataToXdawnCov()
print("j'ai fait le spatial filtering")
m.fit()
print("je viens de finir le fit")

#Partie prédiction sur les données de test

donnees_test=dataLoader.DataLoader(getListOfFiles("data/test"),True)
print("j'ai chargé les données de test")
donnees_test.filtre_Matrice()
print("j'ai filtrer le signal")
pred = m.predict(donnees_test.data)
print("j'ai prédit les labels")
list_id = donnees_test.list_id

#Partie d'écriture de la prédiction sur un fichier
file_sortie = open("submission_kaggle","w")
file_sortie.write("Id,Prediction\n")
for k in range(len(pred)):
    file_sortie.write(str(list_id[k])+","+str(pred[k])+"\n")
file_sortie.close()
