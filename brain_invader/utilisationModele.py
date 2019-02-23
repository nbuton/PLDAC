from genericModele import GenericModele
import pyriemann
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv

def testRiemannMDM(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        m = GenericModele(r,pyriemann.classification.MDM())
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.dataToCov()
        s = m.f1Score()
        result.append([r,s])
    return result

def testRiemannKNN(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        m = GenericModele(r,pyriemann.classification.KNearestNeighbor(n_neighbors=10))
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.dataToCov()
        s = m.f1Score()
        result.append([r,s])
    return result

def testCovPerceptron(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        clf = SGDClassifier(loss="perceptron", eta0=1e-4, learning_rate="constant", penalty=None,tol=1e-1,max_iter=10000,shuffle=True)
        m = GenericModele(r,clf)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.dataToCov()
        m.vectorize()
        s = m.f1Score()
        result.append([r,s])
    return result


def testPerceptronBrut(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        clf = SGDClassifier(loss="perceptron", eta0=1e-4, learning_rate="constant", penalty=None,tol=1e-1,max_iter=10000,shuffle=True)
        m = GenericModele(r,clf)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.vectorize()
        s = m.f1Score()
        result.append([r,s])
    return result

def testKNNBrut(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        neigh = KNeighborsClassifier(n_neighbors=10)
        m = GenericModele(r,neigh)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.vectorize()
        s = m.f1Score()
        result.append([r,s])
    return result

def testPerceptronPaseBas(reactTimeToTest,nbPoint,slider):
    result=[]
    for r in reactTimeToTest:
        clf = SGDClassifier(loss="perceptron", eta0=1e-4, learning_rate="constant", penalty=None,tol=1e-1,max_iter=10000,shuffle=True)
        m = GenericModele(r,clf)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.vectorize()
        m.dataToMoy(nbPoint,slider)
        s = m.f1Score()
        result.append([r,s])
    return result

def testKNNPaseBas(reactTimeToTest,nbPoint,slider):
    result=[]
    for r in reactTimeToTest:
        neigh = KNeighborsClassifier(n_neighbors=10)
        m = GenericModele(r,neigh)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.vectorize()
        m.dataToMoy(nbPoint,slider)
        s = m.f1Score()
        result.append([r,s])
    return result

def testPerceptronTF(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        clf = SGDClassifier(loss="perceptron", eta0=1e-4, learning_rate="constant", penalty=None,tol=1e-1,max_iter=10000,shuffle=True)
        m = GenericModele(r,clf)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.dataToTf()
        m.vectorize(tf=True)
        s = m.f1Score()
        result.append([r,s])
    return result

def testKNNTF(reactTimeToTest):
    result=[]
    for r in reactTimeToTest:
        neigh = KNeighborsClassifier(n_neighbors=10)
        m = GenericModele(r,neigh)
        m.load_data_from_file("data/subject1/Session1/1.gdf")
        m.dataToTf()
        m.vectorize(tf=True)
        s = m.f1Score()
        result.append([r,s])
    return result


def saveResult(name,result):
    myfile = open(name, 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["Time(second)","F1 Score"])
    for r in result:
        wr.writerow(r)


#saveResult("resultats/resultat_riemann_MDM.csv",testRiemannMDM([1,0.1,0.04]))
#saveResult("resultats/resultat_riemann_KNN.csv",testRiemannKNN([1,0.1,0.04]))
#saveResult("resultats/resultat_perceptron_brut.csv",testPerceptronBrut([1,0.1,0.04]))
#saveResult("resultats/resultat_knn_brut.csv",testKNNBrut([1,0.1,0.04]))
#saveResult("resultats/resultat_cov_perceptron.csv",testCovPerceptron([1,0.1,0.04]))
#saveResult("resultats/resultat_passe_bas_perceptron.csv",testPerceptronPaseBas([1,0.1,0.04],10,4))
#saveResult("resultats/resultat_passe_bas_KNN.csv",testKNNPaseBas([1,0.1,0.04],10,4))
#saveResult("resultats/resultat_perceptron_tf.csv",testPerceptronTF([1,0.1,0.04]))
saveResult("resultats/resultat_knn_tf.csv",testKNNTF([1,0.1,0.04]))
