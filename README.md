# Data Scientist Nanodegree

## Deep Learning

## Project: Image Classification with PyTorch

## Table of Contents

1. [Part 1: Developing an AI Application](#p1)
    - [Loading the Data](#load_data)
    - [Label Mapping](#label_map)
    - [Building and Training the Classifier](#build_train)
    - [Testing the Network](#test)
    - [Saving the Checkpoint](#save_ckp)
    - [Loading the Checkpoint](#load_ckp)
    - [Inference for Classification](#infer)
        - [Image Pre-processing](#img_prep)
        - [Class Prediction](#cls_pred)
        - [Sanity Checking](#sanity_ck)
2. [Part 2: Command Line App](#p2)
    - [Specifications](#specs)

## Part 1: Developing an AI Application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, we'll train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells us the name of the flower our camera is looking at. In practice we'll train this classifier in a jupyter notebook, then export it for use in our application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on our dataset
- Use the trained classifier to predict image content

When we've completed this project, we'll have an application that can be trained on any set of labeled images. Here our network will be learning about flowers and end up as a **command line application**. But, what you do with your new skills depends on your imagination and effort in building a data set. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own data set and make something new.

## Part 2: Command Line App