# Plant_Classification
Plant Classification using Artificial Neural Networks.  
Code written in Python 3 , using Tensorflow 2.0 alongside with other libraries for the implementation.

# Dataset 
Dataset required: PlantCLEF 2017 competition dataset.  
We use "trusted" training dataset & test dataset for this project.  
You should first run the "create_structured_test_set.py" and use those structured test data (it changes the way the images are organized in folders).  
Then you create new directory for Training set named `/Data/Train_data/` and Test set named `/Data/Test_data/`.  
Each directory contains 100 classes.  
Those classes must be the same for Train & Test set (e.g. Training and Test folder must contain the class 23). 

# Training 
The training process takes place using the test_train.py.  
This file can be seen as the main file of our code.  
The models that we use is:  
*ResNet50V2  
*MobileNet  
*InceptionV3  
*AlexNet  
The default model is MobileNet.
