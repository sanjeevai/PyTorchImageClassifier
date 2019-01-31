# Data Scientist Nanodegree

## Deep Learning

## Project: Image Classification with PyTorch

## Table of Contents

- [Part 1: Developing an AI Application](#p1)
  - [Loading the Data](#load_data)
  - [Label Mapping](#label_map)
  - [Building and Training the Classifier](#build_train)
  - [Testing the Network](#test)
  - [Save the Checkpoint](#save_ckp)
  - [Loading the Checkpoint](#load_ckp)
  - [Inference for Classification](#infer)
    - [Image Pre-processing](#img_prep)
    - [Class Prediction](#cls_pred)
    - [Sanity Checking](#sanity_ck)
- [Part 2: Command Line App](#p2)
  - [Specifications](#specs)
- [Files](#files)
- [Running](#run)
- [Software and Libraries](#sw_lib)

***

<a id="p1"></a>

## Part 1: Developing an AI Application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I'll train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells us the name of the flower our camera is looking at. I'll train this classifier in a Jupyter Notebook, then export it for use in my application. I'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

- **Load** and **preprocess** the image dataset
- **Train** the image classifier on our dataset
- Use the trained classifier to **predict** image content

When I've completed this project, I'll have an application that can be trained
on any set of labeled images. Here my network will be learning about flowers and
end up as a **command line application**. But, what you do with your new skills
depends on your imagination and effort in building a data set. For example,
imagine an app where you take a picture of a car, it tells you what the make and
model is, then looks up information about it. Go build your own data set and
make something new.

<a id="load_data"></a>

### Loading the Data

Here I'll use `torchvision` to load the data
[[docs]](https://pytorch.org/docs/0.3.0/torchvision/index.html).Data is
present in _flowers_ directory, still you can download it from
[here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)
too.

The dataset is split into three parts: **training, validation** and **testing.**

For the training, I'll apply transformations such as random scaling, cropping
and flipping. This will help the network generalise leading to a better
performance. I'll also make sure the input data is resized to **224x224** as
required by the **pre-trained** networks.

The validation and testing sets are used to measure the model's performance on
data it hasn't seen yet. For this, I won't be applying scaling or rotation
transformations, but will resize then crop the images to the appropriate size.

<a id="label_map"></a>

### Label Mapping

I'll also load in a mapping from **category label** to **category name.** You
can find this in the file `cat_to_name.json`. It's a JSON object which you can
read in with the `json` [module](https://docs.python.org/2/library/json.html).
This will give you a dictionary mapping the integer encoded categories to the
actual names of the flowers.

<a id="build_train"></a>

### Building and Training the Classifier

Now that the data is ready, now I'll build and train the classifier. As usual, I
will use `vgg16` from `torchvision.models` to get the image features. Then I'll
build and train a new feed-forward classifier using those features.

#### Step 1. Load a pre-trained network

Build and train a pre-trained network:
```python
model = models.vgg16(pretrained = True)
```

Freeze the model's parameters so that we don't propagate through them:
```python
for param in model.parameters():
    param.requires_grad = False
```

#### Step 2. Define a new, untrained feed-forward network as a classifier, using ReLU
activations and dropout

```python
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu1', nn.ReLU()),
    ('fc6', nn.Linear(4096, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

model.to(device)
```

The last part will load the model on GPU if you have set it as default in the
starting of the notebook.

#### Step 3. Train the classifier layers using backpropagation using the pre-trained
network to get the features

Evaluation metric:
```python
criterion = nn.CrossEntropyLoss()
```

Optimizer:
```python
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.0001)
```

#### Step 4. Track the loss and accuracy on the validation set to determine the best hyperparameters

<a id="test"></a>

### Testing the Network

It's good practice to test the trained network on test data, images the network
has never seen either in training or validation. This will give a good estimate
for the model's performance on completely new images. Running  the test images
through the network and measuring the accuracy provided an accuracy of 86%.

<a id="save_ckp"></a>

### Save the Checkpoint

Now that the network is trained, I'll save the model so that I can load it later
for making predictions. I also want to save other things such as the mapping of
**classes** to **indices** which I can get from one of the image datasets:
`image_datasets['train'].class_to_idx`. I will attach this to the model as an
attribute which makes inference easier later on.

```python
model.class_to_idx = image_datasets['train'].class_to_idx
```

I'll want to completely rebuild the model later so I can use it for inference,
so I will include any information I need in the checkpoint. 

If I want to load
the model and keep training, I'll need to save the number of epochs as well as
the optimizer state, `optimizer.state_dict`. I'll definitely use this trained
model Part 2, so it is best to save now.

## Part 2: Command Line App

<a id="files"></a>

## Files
<pre>
.
├── Image Classifier Project.ipynb---# BUILD AND TRAIN MODEL FOR PART I
├── assets---------------------------# REFERENCE IMAGES USED IN THE NOTEBOOK
├── cat_to_name.json-----------------# FLOWERS ID TO CLASS LABEL MAPPER
├── flowers--------------------------# DATA DIRECTORY
│   ├── test-------------------------# TEST DATA
│   ├── train------------------------# TRAIN DATA
│   └── valid------------------------# VALIDATION DATA
├── predict.py-----------------------# RUN THIS SCRIPT (WITH EXTERNAL ARGUMENTS)
                                       TO PREDICT ANY LABELLED IMAGE FROM TEST
                                       DIRECTORY
└── train.py-------------------------# RUN THIS SCRIPT (WITH EXTERNAL ARGUMENTS)
                                       TO TRAIN THE MDOEL AND SAVE THE CHECKPOINT
</pre>

<a id="run"></a>

## Running

<a id="sw_lib"></a>

## Software and Libraries

This project uses Python 3.6.3 and the necessary libraries are mentioned in _requirements.txt_