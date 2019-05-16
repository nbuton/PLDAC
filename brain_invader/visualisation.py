import dataLoader
import collections
import numpy as np
import matplotlib.pyplot as plt

def moyennation(l,taille):
    new=[]
    jusqua=int(len(l)/taille)
    for k in range(jusqua):
        new.append(np.mean(l[k*taille:(k+1)*taille]))
    return new

def moyennationWindow(l,taille,decallage):
    new=[]
    jusqua=int(len(l)/decallage)
    for k in range(jusqua):
        new.append(np.mean(l[k*decallage:k*decallage+taille]))
    return new

reactionTime = 1
dataL = dataLoader.DataLoader(["data/subject1/Session1/1.gdf"], reactionTime)


#Visualiser les données
#Deux type de labels 1 et 2
data = dataL.data
labels = dataL.labels
print(len(labels))
collect = collections.Counter(labels)
print("répartition des différentes classes : ")
print(collect)
groupeFlash = labels.reshape((-1,12 ))
"""
On predit la classe majoritaire
classe 1 :
Précision de 0.8333
Rappel de 1
F1 : 2*0.833/1.833 = 0.9091
classe 2 :
Précison : 0
F1 : 0
micro avg : 0.8333
macro avg : 0.45
"""
"""
for g in groupeFlash[:10]:
    nb1 =len(np.argwhere(g==1))
    nb2 =len(np.argwhere(g==2))
    print("["+str(nb1)+","+str(nb2)+"]")
"""
argC1 = np.argwhere(labels==1).reshape((-1))
argC2 = np.argwhere(labels==2).reshape((-1))


dataC1 = data[argC1]
dataC2 = data[argC2]

dataC1Std = np.std(dataC1,axis=0)
dataC2Std = np.std(dataC2,axis=0)
print(dataC2Std.shape)

dataC1Mean = np.mean(dataC1,axis=0)
dataC2Mean = np.mean(dataC2,axis=0)


#On séléctionne que l'éléctrode numero variable elct
elect = 3

#Print somme sample
echantillon = 3
print(data.shape)
for k in range(echantillon):
    print(argC1[k])
    print(argC2[k])
    plt.clf()
    f = plt.figure()
    plt.xlabel("Temps en secondes")
    plt.ylabel("Potentiel de l'éléctrode numero "+str(elect))
    plt.plot(moyennationWindow([d[elect] for d in data[argC1[k]]],40,1),color="red",label="classe 1")
    plt.plot(moyennationWindow([d[elect] for d in data[argC2[k]]],40,1),color="green",label="classe 2")
    plt.legend()
    f.savefig("visualisation/visuel_data_"+str(k)+".pdf", bbox_inches='tight')
#End print sample


dataC1MeanE1 = np.array([d[elect] for d in dataC1Mean])
dataC2MeanE1 = np.array([d[elect] for d in dataC2Mean])

dataC1StdE1 = np.array([d[elect] for d in dataC1Std])
dataC2StdE1 = np.array([d[elect] for d in dataC2Std])

#On moyenne pour que cela sois plus lisible
taille= 10
dataC1MeanE1 = moyennation(dataC1MeanE1,taille)
dataC2MeanE1 = moyennation(dataC2MeanE1,taille)
dataC1StdE1 = moyennation(dataC1StdE1,taille)
dataC2StdE1 = moyennation(dataC2StdE1,taille)

timeList = [k*(1/(reactionTime*int(512/taille))) for k in range(int(512/taille*reactionTime))]
plt.clf()
f = plt.figure()
plt.xlabel("Temps en secondes")
plt.ylabel("Potentiel de l'éléctrode numero "+str(elect))
plt.errorbar(timeList,dataC1MeanE1,dataC1StdE1,color="red",marker='^',linestyle='None',capsize=3,label="classe 1")
plt.errorbar(timeList,dataC2MeanE1,dataC2StdE1,color="green",marker='^',linestyle='None',capsize=3,label="classe 2")
plt.legend()
f.savefig("visualisation/visuel_classe_mean_std.pdf", bbox_inches='tight')
