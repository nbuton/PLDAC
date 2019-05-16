import matplotlib.pyplot as plt
from genericModele import GenericModele
import pyriemann
import csv
import dataLoader
import os

def saveResult(name,result):
    myfile = open(name, 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["Time(second)","F1 Score"])
    for r in result:
        wr.writerow(r)


myFileList = ["data/subject1/Session1/1.gdf"]
reactTimeToTest = [1,0.1,0.04]

mesDonnees = dict()
for r in reactTimeToTest:
    mesDonnees[r]=dataLoader.DataLoader(myFileList,r)

os.mkdir("resultats/rieamannKNN_valeurs_k")
for r,donnees in mesDonnees.items():
    print(r)
    result=[]
    for k in range(1,20,2):
        m = GenericModele(pyriemann.classification.KNearestNeighbor(n_neighbors=k),donnees)
        m.dataToCov()
        s = m.f1Score()
        print([k,s])
        result.append([k,s])
    plt.clf()
    f = plt.figure()
    plt.title("F1 score en fonction de la valeur de k")
    plt.xlabel("Valeur de k")
    plt.ylabel("F1 score")
    plt.plot([re[0] for re in result],[re[1] for re in result])
    name = "resultats/rieamannKNN_valeurs_k/riemannKNN_valeur_de_k_pour_time_"+str(r)
    f.savefig(name+".pdf", bbox_inches='tight')
    saveResult(name+".csv",result)
