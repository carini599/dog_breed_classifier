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

## Project Motivation <a name="motivation"></a>

This is the capstone project of my Nanodegree in "Data Science" at Udacity. In the Jupyter Notebook dog_app.ipynb, I work along several excercises to set up different CNNs(convolutional neural networks). 
First goal was to evaluate the classification of photos of humans and dogs. 
In the next step, I set up a custom CNN to classify dogs according to their breed. 
In Step 4, the Jupiter Notebook guides through upsetting a CNN using transfer learning of the VGG-16-Model.
In Step 5, I set up my own CNN based on transfer learning. 
Based on this model, I finally provide an algorithm, that takes in a photo and classifies it, whether it is a human, a doog, and to which breed it resembles the most.

## File Descriptions <a name="files"></a>

* dog_app.ipynb: Jupiter Notebook, where I work along several excercises to set up a custom and a transfer learning CNN to classify images.
* extract_bottleneck_features.py: function to load files from VGG16 model
* saved_models: Folder with Model results
* dog_images: Folder with images and breeds of dogs
* haarcascades: Folder with model weights of Face classifier
* lfw: Folder with photos of humans

## Results <a name="results"></a>

The custom CNN results in an accuracy of 3.1%. 


## Licensing, Authors, Acknowledgements <a name="licensing"></a>

The Jupiter Notebook with comments and some basic code was provided by Udacity. 
Thanks again to Udacity for pushing me to my limits!
