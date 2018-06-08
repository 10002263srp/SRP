# SRP

This page contains the code that was used in the creation of the SRP.
The Notebooks folder contains the interactive python notebook files with the code that was executed in the creation of the models.
These can also be found at https://drive.google.com/drive/folders/1uk0xlxHZlFP5aczkevVosHoVhZm2IBL6?usp=sharing

The preprocessing folder contains code that was used to vectorise the image data, to link the vectorised images to the labels, and to find the correct hyperparameters for the data. 

The Tensorflow Conversion folder contains the code that was used to convert the Keras hdf5 model to a protobuffer format. 

The hdf5 folder contains code that was used to convert the model to a JSON format, which was then used to create a FHIR resource found at https://simplifier.net/snippet/danieldropuljic/1
