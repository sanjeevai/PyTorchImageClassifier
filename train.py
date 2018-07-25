# Imports here
import argparse
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from PIL import Image

import seaborn as sns
import time
from torch import nn
from torch import optim
import torch
from torch.autograd import Variable
import torch.utils.data as data

from torchvision import datasets, transforms, models
import torchvision

from tqdm import tqdm

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, help='Model\'s Architecture')
parser.add_argument('--checkpoint_path', type=str, help = 'Save trained model to this checkpoint file')
parser.add_argument('--epochs', type=int, help = 'Number of epochs')
parser.add_argument('--data_directory', type=str, help='This directory three folder for training data, testing data, and validation data')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--learning_rate', type=float, help='Learning Rate')

args = parser.parse_args()

print(args)

# this method loads the model

def load_model(arch="vgg16", num_labels=102):
    """
    Args:
    1. arch: It is the architecture which I used for training. Here, it is vgg16
    2. num_label: Number of class labels. Hers there are 102 classes
    """
    
    # load pre-trained model
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        print("Unexpected network architecture")
    
    # freezing the model's parameters so that we don't propagate through them
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('fc6', nn.Linear(4096, num_labels)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    
    return model

# this method trains the model
def train_model(arch='vgg16', epochs = 2, learning_rate = 0.0001, gpu=False, checkpoint_path=''):
    """
    Args:
    1. arch: Architecture of model
    2. epochs: Number of epoch to train for
    3. learning_rate: Learning rate of model. Default set to 0.0001 for better results
    4. gpu: Whether to use GPU or not
    5. checkpoint: Save model checkpoint to this file path
    """
    # use command line arguments
    if args.arch: # architecture
        arch = args.arch
    if args.epochs: # epochs
        epochs = args.epochs
    if args.learning_rate: #learning rate
        learning_rate = args.learning_rate
    if args.gpu: #use GPU
        gpu = args.gpu
    if args.checkpoint_path: # checkpoint path
        checkpoint_path = args.checkpoint_path
    
    print('Pre-trained model architecture:', "vgg16")
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)
    # num_labels = len(train_set.classes)
    model = load_model(arch=arch)

    if gpu and torch.cuda.is_available():
        print('Training on GPU')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Training on CPU')
        device = torch.device("cpu")

    # define metrics and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    st = time.time()
    # train
    for epoch in range(epochs):  # loop over the dataset multiple times
        print("epoch no. {}".format(epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            
            print(i, time.time() - st)

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            print(running_loss)
            # print("running loss is: {}".format(running_loss))
            # if i % 200 == 199:    # print every 200 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

    print('Finished Training in {} seconds'.format(round(time.time() - st, 0)))
    
    model.class_to_idx = train_set.class_to_idx
    # save model to checkpoint file
    if checkpoint_path:
        print("Saving checkpoint here: {}".format(checkpoint_path))
        checkpoint_dict = {'arch':'vgg16',
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.classifier
        }
        
        torch.save(checkpoint_dict, checkpoint_path)

if args.data_directory:
    # Define transforms for the training, validation, and testing data
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms = [train_transforms, test_transforms, valid_transforms]

    # Load the datasets with ImageFolder
    train_set = datasets.ImageFolder(root = "{}/train".format(args.data_directory), transform=train_transforms)
    test_set = datasets.ImageFolder(root = "{}/test".format(args.data_directory), transform=test_transforms)
    valid_set = datasets.ImageFolder(root = "{}/valid".format(args.data_directory), transform=valid_transforms)
    image_datasets = [train_set, test_set, valid_set]
    train_loader = data.DataLoader(train_set, batch_size = 4, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size = 4, shuffle=True)
    valid_loader = data.DataLoader(valid_set, batch_size = 4, shuffle=True)
    dataloaders = [train_loader, test_loader, valid_loader]


    train_model()