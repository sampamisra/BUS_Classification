# BUS_Classification
1. early.py is used for early stopping

2. Train the model by running train_classification_B_SE.py file


3. Test the performance of Alexnet and ResNet model using test_classification_B_SE.py


4. Train the classification layer only using train_classification_B_SE_Ensemble.py by ensembling AlexNet and ResNet models

5. Final models are stored in Model folder

6. Test the final model using test_classification_B_SE_Ensemble.py


crop_and_erase.m is used for cropping B-mode and Strain images from the original images in such a way that they are co-aligned.

The detailed steps of getting B-mode and SE images  from original images are:
1.	Compare B-mode and SE US images pixels by pixels and alignment.
2.	Set the starting point manually (starting point of SE images and B-mode images should be the same).
3.	Divide the image into 2 images (SE and B-mode). 
4.	Erase the circle marks from the B-mode and SE images (as these marks have the same RGB values, we can erase these marks). 
5.	Crop the image from starting point with fixed x and y size (we used 224 x 224). 




