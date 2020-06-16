# Mineral Classification

This project about mineral calssification using image of the mineral and classify it.

Language and Frameworks:  
1. Phyton  
2. Pytorch

The dataset is from Kaggle contains 961 files
[Mineral Identification](https://www.kaggle.com/asiedubrempong/minerals-identification-dataset)

The datasets has 7 labels :  
![Muscovite](https://upload.wikimedia.org/wikipedia/commons/2/26/Muscovite-Albite-122887.jpg)    
1 **Muscovite**: is a hydrated phyllosilicate mineral of aluminium and potassium with formula KAl2(AlSi3O10)(F,OH)2, or (KF)2(Al2O3)3(SiO2)6(H2O). Muscovite is the most common mica, found in granites, pegmatites, gneisses, and schists, and as a contact metamorphic rock or as a secondary mineral resulting from the alteration of topaz, feldspar, kyanite.  

![Chrysocolla](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Chrysocolla-230109.jpg/220px-Chrysocolla-230109.jpg)  
2. **Chrysocolla**: is a hydrated copper phyllosilicate mineral with formula: Cu2−xAlx(H2−xSi2O5)(OH)4·nH2O (x<1)[1] or (Cu,Al)2H2Si2O5(OH)4·nH2O. Associated minerals are quartz, limonite, azurite, malachite, cuprite, and other secondary copper minerals.  

![Quartz](https://geologyscience.com/wp-content/uploads/2018/04/Quartz-and-Hematite-Crystals.jpg)  
3. **Quartz**: is a hard, crystalline mineral composed of silicon and oxygen atoms. Quartz is a defining constituent of granite and other felsic igneous rocks. It is very common in sedimentary rocks such as sandstone and shale. It is a common constituent of schist, gneiss, quartzite and other metamorphic rocks.  

![Quartz](https://geology.com/minerals/photos/bornite.jpg)  
4. **Bornite**: is a sulfide mineral with chemical composition Cu5FeS4. Bornite is an important copper ore mineral and occurs widely in porphyry copper deposits along with the more common chalcopyrite.   

![Pyrite](https://5.imimg.com/data5/KH/IF/ZI/SELLER-10923288/iron-pyrite-crystal-rock-chunks-500x500.jpg)  
5. **Pyrite**: is an iron sulfide with the chemical formula FeS2 (iron(II) disulfide).  

![Malachite](https://images-na.ssl-images-amazon.com/images/I/617z%2Bp25KWL._AC_SX425_.jpg)  
6. **Malachite**: is a copper carbonate hydroxide mineral, with the formula Cu2CO3 (OH)2. Malachite often results from the weathering of copper ores, and is often found with azurite (Cu3(CO3)2(OH)2), goethite, and calcite.   

![Biotite](https://geology.com/minerals/photos/biotite-15.jpg)  
7. **Biotite**: Biotite is a common group of phyllosilicate minerals within the mica group with the approximate chemical formula K(Mg,Fe)3AlSi3O10(F,OH)2.

There is two notebook:  
1. Training notebook
2. Model application using uploaded image using url

From this project I got two models:  
1. My model using 4 convolutional layers and 2 dense layers  
accuracy : 80% in test data
2. VGG16 model and transfers its weight and train the classifier layer  
accuracy : 96% in test data
