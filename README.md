# dog_breed_classifier
Udacity capstone Project: CNN (Convolutional Neural Network) model to predict dog breeds.

## Table of Contents
1.[Installation](#installation)
2.[Project Motivation](#motivation)
3.[File Descriptions](#files)
4.[Results](#results)
5.[Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code is written using Pythons version 3.11.0. All libraries are available within the Anaconda distribution of Python.
The following packages are necessary to run the code of the 
* Jupyter Notebook: dog_app.ipynb
    * opencv-python-headless==4.9.0.80
    * sklearn
    * keras       
    * numpy
    * glob
    * random
    * cv2
    * matplotlib
    * PIL
    * seaborn
    * pandas
    * os

* Web App (Dog App)
    * flask
    * PIL
    * io
    * base64
    * cv2
    * keras
    * numpy

To run the Jupyter Notebook dog_app.ipynb, you need to add the following folders and their content from the Udacity workspace since their content is too large to upload them to GitHub.

* dog_images: Folder with images and breeds of dogs (I downloaded the files from Kaggle using the link: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip, since there was no download option in the Udacity Workspace to download more than one file at a time.)
* lfw: Folder with photos of humans (I downloaded the files from Kaggle using the link: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip, since there was no download option in the Udacity Workspace to download more than one file at a time.)
* bottleneck_features: 
    * DogVGG16Data.npz
    * DogXceptionData.npz

## Project Motivation <a name="motivation"></a>

This is the capstone project of my Nanodegree in "Data Science" at Udacity. In the Jupyter Notebook dog_app.ipynb, I work along several excercises to set up different CNNs(convolutional neural networks). 
First goal was to evaluate the classification of photos of humans and dogs. 
In the next step, I set up a custom CNN to classify dogs according to their breed. 
In Step 4, the Jupiter Notebook guides through upsetting a CNN using transfer learning of the VGG-16-Model.
In Step 5, I set up my own CNN based on transfer learning of the Xception Model. 
Based on this model, I finally provide an algorithm, that takes in a photo and classifies it, whether it is a human, a dog, and to which breed it resembles the most.

To make use of this classification algorithm, I finally set up a web app using flask, which takes in a custom photo and classifies it.
You can find further information about the development of my dog classification app on my blog using the link: https://breuerei.de/classify-your-dogs-breed-or-find-your-barking-twin-my-journey-into-neural-networks/ 

## File Descriptions <a name="files"></a>

* dog_app.ipynb: Jupiter Notebook, where I work along several excercises to set up a custom and a transfer learning CNN to classify images.
* extract_bottleneck_features.py: function to load files from different pretrained CNNs
* saved_models: Folder with Model results
* dog_images: Folder with images and breeds of dogs 
    (I downloaded the files from Kaggle using the link: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip, since there was no download option in the Udacity Workspace to download more than one file at a time.)
* haarcascades: Folder with model weights of face classifier
* lfw: Folder with photos of humans (I downloaded the files from Kaggle using the link: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip, since there was no download option in the Udacity Workspace to download more than one file at a time.)
* bottleneck_features: model data of VGG16 and Xception trained on dog images
* dog_app : Folder with Web App Code
    * app
        * static: photos which show examples of dog breeds
        * templates: html files for dog_app (master.html as start page and go.html as classification page)
        * dog_app_functions.py: collection of functions to return model results that are used by run.py
        * extract_bottleneck_features.py: function to load files from different pretrained CNNs
        * run.py: central code for dog app, based on flask

    * models: Xception based model trained on dog images
* images: images used in dog_app.ipynb and test images for classification. 

## Results <a name="results"></a>

For results the results of my capstone project, please refer to my blog post on https://breuerei.de/classify-your-dogs-breed-or-find-your-barking-twin-my-journey-into-neural-networks/ and check out my Dog Classification App on
https://breuerei.de/dog_app

## Screenshots Web App

![Dog Breed Classifier](images/screenshot_dog_app_1.jpg)

![Dog Breed Classifier](images/screenshot_dog_app_2.jpg)


## Licensing, Authors, Acknowledgements <a name="licensing"></a>

The Jupiter Notebook with comments and some basic code was provided by Udacity. 
Thanks again to Udacity for pushing me to my limits!
