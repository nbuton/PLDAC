import matplotlib.pyplot as plt
from genericModele import GenericModele
import pyriemann
import csv

def saveResult(name,result):
    myfile = open(name, 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["Time(second)","F1 Score"])
    for r in result:
        wr.writerow(r)


reactTimeToTest = [1,0.1,0.04]

for r in reactTimeToTest:
    print(r)
    result=[]
    for k in range(1,50,2):
        m = GenericModele(r,pyriemann.classification.KNearestNeighbor(n_neighbors=k))
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.dataToCov()
        s = m.f1Score()
        result.append([k,s])
    plt.clf()
    f = plt.figure()
    plt.plot([re[0] for re in result],[re[1] for re in result])
    name = "resultats/riemannKNN_valeur_de_k_pour_time_"+str(r)
    f.savefig(name+".pdf", bbox_inches='tight')
    saveResult(name+".csv",result)
