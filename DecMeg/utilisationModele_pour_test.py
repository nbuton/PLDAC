from genericModele import GenericModele
from model_all_pipeline import Model_all_pipeline
import pyriemann
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv
import os
import dataLoader
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv1D,Dropout,MaxPooling1D,GlobalAveragePooling1D
from keras import backend as K
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import XdawnCovariances
from pyriemann.classification import MDM
from sklearn.linear_model import LogisticRegression


def testRiemannMDM(mult_donnees,mesDonnees_test):
    print("debut test riemann MDM")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        m = GenericModele(pyriemann.classification.MDM(),donnees)
        m.dataToCov()
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def testRiemannMDMPlusXdawn(mult_donnees,mesDonnees_test):
    print("debut test riemann MDM")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        m = GenericModele(pyriemann.classification.MDM(),donnees)
        m.dataToXdawnCov()
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def test_all_pipeline(mult_donnees,mesDonnees_test):
    print("debut test all pipeline")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        clf = LogisticRegression()
        m = Model_all_pipeline(clf,donnees)
        m.fit_representation()
        new_data = m.predict_representation(donnees.data)
        print(new_data.shape)
        new_labels = m.concat_labels(donnees.labels)
        m.fit(new_data,new_labels)
        pred = m.predict(new_data)
        print(pred)
        print(pred.shape)
        s = m.f1Score(new_data,new_labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result



def testRiemannKNN(mult_donnees,mesDonnees_test):
    print("debut test riemann KNN")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        m = GenericModele(pyriemann.classification.KNearestNeighbor(n_neighbors=3),donnees)
        m.dataToCov()
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result


def testCovSVM(mult_donnees,mesDonnees_test):
    print("debut test cov SVM")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        clf = SVC(gamma='auto',probability=True,max_iter=100,verbose=1)
        m = GenericModele(clf,donnees)
        m.dataToCov()
        m.vectorize()
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result


def testSVMBrut(mult_donnees,mesDonnees_test):
    print("debut test SVM brut")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        clf = SVC(gamma='auto',probability=True,max_iter=100,verbose=1)
        m = GenericModele(clf,donnees)
        m.vectorize()
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def testKNNBrut(mult_donnees,mesDonnees_test):
    print("debut test knn brut")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        neigh = KNeighborsClassifier(n_neighbors=3)
        m = GenericModele(neigh,donnees)
        m.vectorize()
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def testSVMPaseBas(nbPoint,slider,mult_donnees,mesDonnees_test):
    print("debut test passe bas SVM")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        clf = SVC(gamma='auto',probability=True,max_iter=100,verbose=1)
        m = GenericModele(clf,donnees)
        m.vectorize()
        m.dataToMoy(nbPoint,slider)
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def testKNNPaseBas(nbPoint,slider,mult_donnees,mesDonnees_test):
    print("debut test passe bas knn")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        neigh = KNeighborsClassifier(n_neighbors=3)
        m = GenericModele(neigh,donnees)
        m.vectorize()
        m.dataToMoy(nbPoint,slider)
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def testSVMTF(mult_donnees,mesDonnees_test):
    print("debut test SVM tf")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        clf = SVC(gamma='auto',probability=True,max_iter=100,verbose=1)
        m = GenericModele(clf,donnees)
        m.dataToTf()
        m.vectorize(tf=True)
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result

def testKNNTF(mult_donnees,mesDonnees_test):
    print("debut test knn tf")
    result=[]
    for r,donnees in mult_donnees.items():
        print("r = "+str(r))
        neigh = KNeighborsClassifier(n_neighbors=3)
        m = GenericModele(neigh,donnees)
        m.dataToTf()
        m.vectorize(tf=True)
        m.fit()
        s = m.score(mesDonnees_test[r].data,mesDonnees_test[r].labels)
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result


def testCNN1DKeras(mult_donnees,mesDonnees_test):
    print("debut test conv1D")
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


    def create_model():
        # create model
        model = Sequential()
        model.add(Conv1D(4, 10, strides=2, padding='same', activation='relu'))
        model.add(Dropout(0.4))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(40, 5, strides=2, padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['binary_accuracy'])
        return model
    result=[]
    for r,donnees in mult_donnees.items():
        print(donnees.data.shape)
        print("r = "+str(r))
        model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=1000)
        m = GenericModele(model,donnees)
        s = m.f1Score(mode="proba")
        result.append([r,s])
    print("reactionTime,f1score : "+str(result))
    return result





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

def keep_only_gdf_file(fileList):
    new_fileList=[]
    for f in fileList:
        if(f[-3:]=="gdf"):
            new_fileList.append(f)
    return new_fileList

def keep_one_file_per_subject(fileList):
    new_fileList = []
    for f in fileList:
        if("Session1" in f and "1.gdf" in f ):
            new_fileList.append(f)
    return new_fileList

def saveResult(name,result):
    myfile = open(name, 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["Time(second)","F1 Score"])
    for r in result:
        wr.writerow(r)

nomTest = "test_all_other_model_1train_1test"

#myFileList=["data/train/train_subject1.mat"]
#myFileList = getListOfFiles("data/train/")[:4]

mesDonnees = dict()
mesDonnees[1.5]=dataLoader.DataLoader(["data/train/train_subject01.mat"],concatenate=True)

mesDonnees_test = dict()
mesDonnees_test[1.5]=dataLoader.DataLoader(["data/train/train_subject02.mat"],concatenate=True)
#print("j'ai enlevé le filtre passe bande car les matrices netaits plus définit positives apres")
#mesDonnees[1.5].filtre_Matrice()
print(len(mesDonnees[1.5].data))
print(len(mesDonnees[1.5].labels))

#saveResult("resultats/"+nomTest+"/resultat_riemann_MDM_xDawn.csv",test_all_pipeline(mesDonnees))
#print("Attention nouveau pre processing uniquement pour pyriemann")
#saveResult("resultats/"+nomTest+"/resultat_riemann_MDM_xDawn.csv",testRiemannMDMPlusXdawn(mesDonnees))
os.mkdir("resultats/"+nomTest)

saveResult("resultats/"+nomTest+"/resultat_riemann_MDM.csv",testRiemannMDM(mesDonnees,mesDonnees_test))

#saveResult("resultats/"+nomTest+"/resultat_riemann_KNN.csv",testRiemannKNN(mesDonnees))

saveResult("resultats/"+nomTest+"/resultat_SVM_brut.csv",testSVMBrut(mesDonnees,mesDonnees_test))

#saveResult("resultats/"+nomTest+"/resultat_knn_brut.csv",testKNNBrut(mesDonnees))

#saveResult("resultats/"+nomTest+"/resultat_cov_SVM.csv",testCovSVM(mesDonnees))
try :
    saveResult("resultats/"+nomTest+"/resultat_passe_bas_SVM.csv",testSVMPaseBas(10,4,mesDonnees,mesDonnees_test))
except:
    print("Erreur passe bas SVM")
#saveResult("resultats/"+nomTest+"/resultat_passe_bas_KNN.csv",testKNNPaseBas(10,4,mesDonnees))
try:
    saveResult("resultats/"+nomTest+"/resultat_SVM_tf.csv",testSVMTF(mesDonnees,mesDonnees_test))
except:
    print("Erreur tf SVM")
#saveResult("resultats/"+nomTest+"/resultat_knn_tf.csv",testKNNTF(mesDonnees))
try:
    saveResult("resultats/"+nomTest+"/resultat_conv1D.csv",testCNN1DKeras(mesDonnees,mesDonnees_test))
except:
    print("Erreur conv1D SVM")
