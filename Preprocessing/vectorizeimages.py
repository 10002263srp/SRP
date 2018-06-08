"""
Vectorize a train image set of 32126 images 
"""
#Python image library that vectorises images
from PIL import Image
import numpy as np
import os
import pandas as pd

trainLabels = pd.read_csv("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//trainLabels.csv")

listing = os.listdir("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train") 

#Remove the labels file from the count
listing.remove("trainLabels.csv")

#List number of training images
np.size(listing)


#input image dimensions limited to input for VGGNet model later on
img_rows, img_cols = 224, 224

#Vectorise images 
immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//" + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//" + file)
    img = im.resize((img_rows,img_cols))
    immatrix.append(np.array(img))
    
#Cast vectors to numpy arrays
immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)

#Save created immatrix, imlabel matrices for later use
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//imlabel' ,imlabel)
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//immatrix',immatrix)



"""
Vectorize a test image set of 2000 images 
"""

#Python image library that vectorises images
from PIL import Image
import numpy as np
import os
import pandas as pd

trainLabels = pd.read_csv("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//labels.csv")

listing = os.listdir("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test") 

#Remove the labels file from the count
listing.remove("labels.csv")

#List number of training images
np.size(listing)


#input image dimensions limited to input for VGGNet model later on
img_rows, img_cols = 224, 224

#Vectorise images 
immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//" + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//" + file)
    img = im.resize((img_rows,img_cols))
    immatrix.append(np.array(img))
    
#Cast vectors to numpy arrays
immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)

#Save created immatrix, imlabel matrices for later use
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//imlabel_test' ,imlabel)
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//immatrix_test',immatrix)


"""
Vectorize a validation image set of 1000 images 
"""
#Python image library that vectorises images
from PIL import Image
import numpy as np
import os
import pandas as pd

trainLabels = pd.read_csv("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//vallabels.csv")
listing = os.listdir("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation") 
listing.remove("vallabels.csv")
np.size(listing)

img_rows, img_cols = 224, 224


immatrix = []
imlabel = []

for file in listing:
    base = os.path.basename("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//" + file)
    fileName = os.path.splitext(base)[0]
    imlabel.append(trainLabels.loc[trainLabels.image==fileName, 'level'].values[0])
    im = Image.open("C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//" + file)
    img = im.resize((img_rows,img_cols))
    immatrix.append(np.array(img))
    
#Cast vectors to numpy arrays
immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)

np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//imlabel_val' ,imlabel)
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//immatrix_val',immatrix)
