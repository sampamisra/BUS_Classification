from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils import data
import glob
import os
from sklearn.model_selection import KFold, GroupKFold
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl
data_dir1 = "./BW"
data_dir2 = "./Strain"
num_classes = 2
batch_size =64
num_epochs = 300
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA[1].fc = nn.Identity()
        self.modelB[1].classifier[6] = nn.Identity()        
        # Create new classifier
        self.classifier = nn.Linear(4608, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.classifier(F.relu(x))
        return x

# Train your separate models
# ...
# We use pretrained torchvision models here
modelA = nn.Sequential(
nn.Conv2d(in_channels=4,out_channels=3,kernel_size=(1,1)),
models.resnet18(pretrained=True))
num_ftrs = modelA[1].fc.in_features
modelA[1].fc = nn.Linear(num_ftrs, 2)
modelB = nn.Sequential(
nn.Conv2d(in_channels=4,out_channels=3,kernel_size=(1,1)),
models.alexnet(pretrained=True))
num_ftrs = modelB[1].classifier[6].in_features
modelB[1].classifier[6] = nn.Linear(num_ftrs, 2)
for param in modelA.parameters():
    param.requires_grad_(False)
for param in modelB.parameters():
    param.requires_grad_(False)
modelA.load_state_dict(torch.load('Model\checkpoint_resnet_3.pt'))
modelB.load_state_dict(torch.load('Model\checkpoint_alexnet_3.pt'))
from early import EarlyStopping
if __name__ == '__main__':
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i ):
        model.to(device) 
        epochs = num_epochs
        valid_loss_min = np.Inf
        train_losses = []
    # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 
        train_acc, valid_acc =[],[]
        steps=0
        #valid_acc =[]
        best_acc = 0.0
        import time
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(epochs):    
            start = time.time()
            #scheduler.step()
            model.train()
            total_train = 0
            correct_train = 0
            for inputs, labels in train_loader:
                steps+=1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)                
                optimizer.zero_grad()                
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                _, predicted = torch.max(logps.data, 1)
                total_train += labels.nelement()
                correct_train += predicted.eq(labels.data).sum().item()
                train_accuracy = correct_train / total_train
                model.eval()
               
            with torch.no_grad():
                accuracy = 0
                for inputs, labels in valid_loader:
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    valid_losses.append(loss.item())
        # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
             
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            valid_acc.append(accuracy/len(valid_loader)) 
            train_acc.append(train_accuracy)
        
        # calculate average losses
            
            valid_accuracy = accuracy/len(valid_loader)          
            
            # print training/validation statistics 
            print(f"Epoch {epoch+1}/{epochs}.. ")
            print('Training Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))
            train_losses = []
            valid_losses = []        
            if valid_accuracy > best_acc:
                best_acc = valid_accuracy
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
           
        print('Best val Acc: {:4f}'.format(best_acc*100))  
        torch.save(model.state_dict(), "./Model/checkpoint_Ensemble_{0}.pt".format(i))
        print("model saved")
        
        return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc
class TrainDataset(Dataset) :
    def __init__(self, data_dir1, data_dir2, transform=None) :
        self.BW= sorted(glob.glob(os.path.join(data_dir1,'*','*','*')))
        self.Strain= sorted(glob.glob(os.path.join(data_dir2,'*','*','*')))
        self.transform = transform
    def __getitem__(self, index) :
        data_path_BW = self.BW[index]
        image_BW= Image.open(data_path_BW)
        data_path_Strain = self.Strain[index]
        image_Strain= Image.open(data_path_Strain)
        res = transforms.Resize((224, 224))
        image_BW  = res(image_BW)
        image_Strain  = res(image_Strain)
        image_BW=torchvision.transforms.functional.to_tensor(image_BW)
        image_Strain=torchvision.transforms.functional.to_tensor(image_Strain)
        image= torch.cat((image_BW, image_Strain), dim=0) 

        if data_path_BW.split("\\")[-3] == "Benign" :
            label = 0
        elif data_path_BW.split("\\")[-3] == "Malignant" :
            label = 1
        else :
            raise Exception('invalid path')
        #patient = data_path.split("\\")[-2]
        #patient_num = (int) (patient[ patient.find('(')+1 : patient.find(')') ]) + patient_class        
        return image, label

    def __len__(self) : 
        length = len(self.BW)
        return length      
    def grouplist(self):
        glist = []
        for data_path_BW in self.BW :
            patient = data_path_BW.split("\\")[-2]

            if data_path_BW.split("\\")[-3] == "Benign" :
                patient_class = 1000
            elif data_path_BW.split("\\")[-3] == "Malignant" :
                patient_class = 2000
            patient_num = (int) (patient[ patient.find('(')+1 : patient.find(')') ]) + patient_class
            glist.append(patient_num)
        return glist
feature_extract = True
sm = nn.Softmax(dim = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 50
criterion = nn.CrossEntropyLoss()
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])
merge_data = TrainDataset(data_dir1 + "/Train", data_dir2 + "/Train", transform=train_transforms)
group_list = merge_data.grouplist()
fold_counts= 5
kfold = GroupKFold(n_splits=fold_counts)
num_workers = 0
#
#--------------------------------------------------------------
for i, (train_index, validate_index) in enumerate(kfold.split(merge_data, groups=group_list)):
    # print("train index:", train_index, "validate index:", validate_index)
    # for i in validate_index :
    #     print(merge_data.all_data[i])
    trainPaitients =[]
    valPaitients= []
    for ti in train_index:
        tPatient = group_list[ti]
        if tPatient not in trainPaitients :
            trainPaitients.append(tPatient)
    for vi in validate_index:
        vPatient = group_list[vi]
        if vPatient not in valPaitients :
            valPaitients.append(vPatient)
    train = torch.utils.data.Subset(merge_data, train_index)
    validation = torch.utils.data.Subset(merge_data, validate_index)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    print("Number of Samples in Train: ",len(train))
    print("Number of Patients in Train : ",len(trainPaitients))
    print(sorted(trainPaitients))
    print("Number of Samples in Valid: ",len(validation))
    print("Number of Patients in Valid : ",len(valPaitients))
    print(sorted(valPaitients))
    # patient 1XXX : COVID XXXth Patient (2 : Healthy, 3: Other)
    model = MyEnsemble(modelA, modelB)
    #print(model)
    optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    model, train_loss, valid_loss, train_acc, valid_acc=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i)
