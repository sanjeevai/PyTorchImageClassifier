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

parser.add_argument('--arch', type=str, help='Model\'s Architecture. This model was trained using vgg16 and vgg19 architecture')
parser.add_argument('--data_directory', type=str, help='This directory contains three folders for training data, testing data, and validation data')
parser.add_argument('--learning_rate', type=float, help='Learning Rate')
parser.add_argument('--epochs', type=int, help = 'Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--checkpoint_path', type=str, help = 'Save trained model to this checkpoint file')
parser.add_argument('--image_path', type=str, help='Path for image file which will be used for prediction')
parser.add_argument('--label_file', type=str, help='JSON file containing mapping of number to labels')
parser.add_argument('--top_k', type=int, help='Return top k predictions')

args = parser.parse_args()

# Assume that we are on a CUDA machine, then this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# this method loads the model
def load_model(arch="vgg16", num_labels=102):
    # load pre-trained model
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
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

# this method performs validation on validation data
def validation(model, valid_loader, criterion):
    valid_loss = 0
    accuracy = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

# this method trains the model
def train_model(arch='vgg16', epochs = 2, learning_rate = 0.0001, gpu=False, checkpoint_path=''):
    # use command line arguments
    if args.arch:
        arch = args.arch
    if args.epochs:
        epochs = args.epochs
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.gpu:
        gpu = args.gpu
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    
    print('Network architecture:', arch)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)
    num_labels = len(train_set.classes)
    model = load_model(arch=arch)

    if gpu and torch.cuda.is_available():
        print('Training on GPU')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Training on CPU')
        device = torch.device("cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    steps = 0
    running_loss = 0
    print_every = 800
    st = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            steps += 1
            print(i)

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
            if steps % print_every == 0:
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion)
                print("Epoch:{}/{}".format(epoch + 1, epochs),
                "Training Loss: {:.3f}".format(running_loss/print_every),
                "Validation Loss: {:.3f}".format(valid_loss/len(valid_loader)),
                "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                running_loss = 0.0
                
    print('Finished Training in {} seconds'.format(round(time.time() - st, 0)))
    
    model.class_to_idx = train_set.class_to_idx
    
    if checkpoint_path:
        print("Saving checkpoint here: {}".format(checkpoint_path))
        if arch == 'vgg16':
            checkpoint_dict = {'arch':'vgg16',
                               'class_to_idx': model.class_to_idx,
                               'state_dict': model.state_dict(),
                               'classifier': model.classifier
                              }
        elif arch == 'vgg19':
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