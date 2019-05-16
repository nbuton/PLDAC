#Pour l'instant inutilisé
class CreateFileList():
    def __init__():
        pass
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
