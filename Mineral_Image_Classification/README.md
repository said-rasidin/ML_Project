# Mineral Classification

This project about mineral calssification using image of the mineral and classify it.

Language and Frameworks:  

Phyton
Pytorch

The dataset is from Kaggle contains 961 files
[Mineral Identification](https://www.kaggle.com/asiedubrempong/minerals-identification-dataset)

The datasets has 7 labels :
1. biotite
2. bornite
3. chrysocolla
4. malachite  
5. muscovite  
6. pyrite  
7. quartz  

There is two notebook:  
1. Training notebook
2. Model application using uploaded image using url

From this project I got two models:  
1. My model using 4 convolutional layers and 2 dense layers  
accuracy : 80% in test data
2. VGG16 model and transfers its weight and train the classifier layer  
accuracy : 96% in test data
