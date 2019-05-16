import os
import csv

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


def keep_only_result_file(fileList):
    new_fileList=[]
    for f in fileList:
        if("resultat" in f):
            new_fileList.append(f)
    return new_fileList

def filter_file(filtre,fileList):
    new_fileList=[]
    for f in fileList:
        if(filtre in f):
            new_fileList.append(f)
    return new_fileList

def saveResult(name,result):
    myfile = open(name, 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["nom du fichier","1 seconde","0.1 seconde","0.04 seconde"])
    for r in result:
        wr.writerow(r)


testName = "test_par_groupe_12"
fileList = getListOfFiles("resultats/"+testName+"/")
fileList = keep_only_result_file(fileList)
print(len(fileList))

list_finale=[]
for f in fileList:
    print(f)
    tmp=[f]
    with open(f) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            if(row_count != 0):
                tmp.append(row[1])
            row_count+=1
        list_finale.append(tmp)
print(len(list_finale))
saveResult("resultats/"+testName+"/concat_file.csv",list_finale)
