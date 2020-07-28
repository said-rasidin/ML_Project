# Machine Learning Project
 
This is where I place my personal projects, these are several projects that I have finished so far:

## Computer Vision   

### 1. [Semantic Segmentation in Seismic Images](https://github.com/said-rasidin/ML_Project/tree/master/Seismic%20Segmentation)   
This is semantic segmentation project to delineate salt bodies in seismic images with U-Net Architecture with Resnet18 and Resnet34. Salt can be a tricky process to image in seismic. To better interpret salt bodies in seismic, geologic information needs to be integrated in order to accurately image salt. Salt and sediment interactions can cause reservoir traps and hydrocarbon traps.  
<p align="center">
<img src="./Seismic%20Segmentation/predic1.png" width="720">   
</p>    

### 2. [Drone Aerial View Segmentation](https://github.com/said-rasidin/ML_Project/tree/master/Drone%20Segmentation)    
How to teach drone to see what is below and segment the object with high resolution. Teaching drone to see is quite challenges due to bird’s eye view and most of pre-trained models are trained in normal images we see (point of view) in daily basis (ImageNet, PASCAL VOC, COCO). In this project I want to experiment how to train drone datasets, the aims are:
- Model that light weight (less parameters)
- High score (I hope so)
- Fast inference latency.  
<p align="center">
<img src="./Drone%20Segmentation/images/inference1.png" width="1020">
<img src="./Drone%20Segmentation/images/inference2.png" width="1020">
</p>   

### 3. [Mineral Image Classification](https://github.com/said-rasidin/ML_Project/tree/master/Mineral_Image_Classification)   
I classify 7 mineral classes and produce robust model that can be used to classify minerals, I've tried it using mineral images from Google. Mineral klasifikasi merupakan hal yang biasa bagi seorang geologist, namun jumlah mineral yang sangat banyak membuat tidak semua geologist paham satu-satu termasuk saya (jadi ini berlatar belakang pribadi), oleh karena itu penggunaan deep learning dan computer vision bisa sangat bermanfaat dalam mengklasifikasi mineral langsung dari foto sampel lapangan.
<p align="center">
<img src="./Mineral_Image_Classification/Screenshot%20(65).png" width="480">   
</p>   

### 4. [COVID-19 X-ray Classification](https://github.com/said-rasidin/ML_Project/tree/master/COVID-19%20X-ray)    
This project aimed to identify covid-19 from x-ray images as early detection to help physician   
<p align="center">
<img src="./COVID-19%20X-ray/resnet18_predict.png" width="480"> 
</p>    

## Natural Language Processing   

### 5. [Sentiment Analysis COVID-19](https://github.com/said-rasidin/ML_Project/tree/master/Sentiment%20Analysis%20Covid-19)   
Classifying sentiment from twitter how people reaction in COVID-19 situation (private datasets)   
<p align="center">
<img src="./Sentiment%20Analysis%20Covid-19/words_cloud_neg.png" width="360"> 
</p>    

### 6. [Text Extraction](https://github.com/said-rasidin/ML_Project/tree/master/Text%20Extraction)   
Save your reading time while machine extracts important sentences and make summary from PDF file, using TF-IDF as weight for each word in sentences and return the highes score as summary and save your time.  

## Other topics   

### 7. [Air Quality Data Exploration](https://github.com/said-rasidin/ML_Project/tree/master/Air_Quality)    
Simple tabular data exploration, and build simple linear model to predict air quality index   

### 8. [Feed Forward Neural Network](https://github.com/said-rasidin/ML_Project/tree/master/Neural%20Net)   
Build Deep Neural Network architecture to predict Housing Price and classifying Fashion MNIST datasets   

### 9. [Land Cover Classification from NDVI Value](https://github.com/said-rasidin/ML_Project/tree/master/land_cover_NDVI)
Data Exploration and do some features selection to classify land cover base on its NDVI values    
