import numpy as np
from keras.utils import np_utils

#-----------------------------------

immatrix = np.load('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//immatrix.npy')
imlabel = np.load('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//imlabel.npy')

(X_train, y_train) = (immatrix, imlabel)

#Scale data to fall within [0, 1] interval
X_train = X_train.astype('float32')
X_train /= 255

y_train = np_utils.to_categorical(y_train, 5)

np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//immatrix.npy', immatrix[:9999])
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//train//imlabel.npy', imlabel[:9999])

#---------------------------------
immatrix_test = np.load('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//immatrix_test.npy')
imlabel_test = np.load('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//imlabel_test.npy')


(X_test, y_test) = (immatrix_test, imlabel_test)

X_test = X_test.astype('float32')
X_test /= 255

y_test = np_utils.to_categorical(y_test, 5)

np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//immatrix_test.npy', immatrix_test[:1999])
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//test//imlabel_test.npy', imlabel_test[:1999])

#----------------------

immatrix_val = np.load('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//immatrix_val.npy')
imlabel_val = np.load('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//imlabel_val.npy')

(X_val, y_val) = (immatrix_val, imlabel_val)

X_val = X_test.astype('float32')
X_val /= 255

y_val = np_utils.to_categorical(y_val, 5)

np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//immatrix_val.npy', immatrix_val[:999])
np.save('C://Users//Daniel//Desktop//Code//Retinopathy//fttl//data//validation//imlabel_val.npy', imlabel_val[:999])

#----------------------
print('Shape of X_train:',X_train.shape)
print('Shape of y_train:', y_train.shape)

print('Shape of X_test:',X_test.shape)
print('Shape of y_train:', y_test.shape)

print('Shape of X_val:',X_val.shape)
print('Shape of y_val:', y_val.shape)