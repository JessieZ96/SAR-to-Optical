import argparse
import torch
import numpy as np
import math
# from options.train_options import TrainOptions
# from data1 import create_dataset
import pdb
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse

from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')
args = parser.parse_args()

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, model_path, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3hhhh
        Be careful, expects (2hhhh99,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary netmodel
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary netmodel
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# model name
model_name = "resnet18"
Use_pretrained = True

# classes
num_classes = 5
feature_extract = True
f = open("./confusion_matrix/{}_{}_confusion_matrix.txt".format(model_name, Use_pretrained), "w+")
 

# model path
model_path = "./checkpoints/{}_{}/best".format(model_name, Use_pretrained)
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, model_path, use_pretrained=Use_pretrained)
data_dir = r"./data/sar"

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['val']}
# print(dataloaders_dict.items)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model_ft.to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()
# pdb.set_trace()
if num_classes == 2:

    TP = TN = FP = FN = 0
    result =  np.empty(shape=[0, 2])
    for inputs, labels in dataloaders_dict['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        prob = model(inputs)
        pre = (prob[:, 0] < prob[:, 1])

        TP += ((pre == 1) & (labels == 1)).cpu().sum().numpy()
        TN += ((pre == 0) & (labels == 0)).cpu().sum().numpy()
        FP += ((pre == 0) & (labels == 1)).cpu().sum().numpy()
        FN += ((pre == 1) & (labels == 0)).cpu().sum().numpy()
    # pdb.set_trace()

    p = float(TP / (TP + FP))
    r = float(TP / (TP + FN))
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

elif num_classes > 2:
    Pre = np.empty((0))
    Labels = np.empty((0))
    for inputs, labels in dataloaders_dict['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if "inception" in model_name:
            prob = model(inputs)
        else:
            prob = model(inputs)
        # pdb.set_trace()
        pre = np.argmax(prob.cpu().detach().numpy(), axis=1)
        Pre = np.append(Pre, pre)
        Labels = np.append(Labels, labels.cpu().detach().numpy())
    
    CM = confusion_matrix(Labels, Pre)
    all_CM = np.sum(CM)
    s = 0
    for i in range(CM.shape[0]):
        s += CM[i][i]
    


    print(s / all_CM)
    f.write("\n")
    # f.write("The {} epoch: \n".format(i))
    f.write("acc: {}\n".format(s / all_CM))
    for j in CM:
        f.write('{} {} {} {} {}\n'.format(j[0], j[1], j[2], j[3], j[4]))
    print(CM)
    # pdb.set_trace()

f.close()
