# Acute-Brain-Infarct-Location-Classification-using-CNN
A Deep-Learning project to detect Acute infarcts in different brain locations using Keras and Tensorflow libraries.

## 1. Data Preprocessing
- The **CLEANED_DATA** folder contains sorted (according to the location of infarct) brain MRI images.
- The **DWI_DATA** folder further contains DWI_train and DWI_test subfolders. All the images are ***cropped and resized to (516 x 444) pixels***.
   -  **DWI_train:** It has all **46 folders** *(all unique locations)*. This has to be used as the training data with the respective folder's name (location itself) as a label for the same. 
   -  **DWI_test** : It has **4 folders** *(only repeated locations)*. This has to be used as the test data with the respective folder's name (location itself) as a label for the same. 
