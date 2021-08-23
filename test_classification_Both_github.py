from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
import glob
import os
import csv
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl
data_dir1 = "./BW" #B-mode images Directory
data_dir2 = "./Strain" #Strain images Directory
test_dir1=data_dir1 + '/Test'
test_dir2=data_dir2 + '/Test'
num_classes = 2
batch_size =64
##...............For ResNet18 Model.......................##
# model = nn.Sequential(
# nn.Conv2d(in_channels=4,out_channels=3,kernel_size=(1,1)),
# models.resnet18(pretrained=True))
# num_ftrs = model[1].fc.in_features
# model[1].fc = nn.Linear(num_ftrs, 2)
# #print(model[1].layer3[0].conv2.weight)
# model.load_state_dict(torch.load('Model/checkpoint_resnet_3.pt'))
##...............For Alexnet Model.......................##
model = nn.Sequential(
nn.Conv2d(in_channels=4,out_channels=3,kernel_size=(1,1)),
models.alexnet(pretrained=True))
num_ftrs = model[1].classifier[6].in_features
model[1].classifier[6] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('Model/checkpoint_alexnet_3.pt'))
if __name__ == '__main__':    
    def test(model, criterion):        
        model.to(device) 
        loss_epoch = 0
        accuracy_epoch = 0
        model.eval()
        pwcm = torch.zeros(2, 2)
        pred = []
        true = []
        soft = []
        patient_num = []
        vote=[{},{},{}]
        real = {}

        for step, (x, y, groups) in enumerate(test_loader):
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)            
            # for majority voting
            softmax = torch.nn.Softmax(dim=1)
            # s = softmax(outputs).detach().numpy() 
            # for i in range(len(s)):
            #     soft.append(s[i])
            predicted = outputs.argmax(1)
            preds = predicted.cpu().numpy()
            labels = y.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            
            # voting
            for t, p, pnt in zip( y.view(-1), predicted.view(-1), groups.view(-1)) :
                pn = pnt.item()

                if pn not in patient_num:
                    patient_num.append(pn)
                    vote[0][pn] = 0
                    vote[1][pn] = 0
                    real[pn] = t.long()
                
                vote[p.long()][pn] += 1


            labels = np.reshape(labels, (len(preds), 1))
            for i in range(len(preds)):
                pred.append(preds[i][0].item())
                true.append(labels[i][0].item())            
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            loss_epoch += loss.item()
        
        patient_pred = []
        patient_true = []
        #check the voting
        for pn in patient_num:
            voteList = [vote[i][pn] for i in range(2)]
            #print(voteList[0])
            # if voteList[0] == max(voteList) :
            #     p=0
            # elif voteList[1] == max(voteList) :
            #     p=1  
            # else:
            #     raise Exception()    
            p = voteList.index(max(voteList))
            if voteList[0]== voteList[1]:
                p=1
            ##print(p)
            pwcm[real[pn],p] += 1
            patient_true.append(int(real[pn]))
            patient_pred.append(int(p))

        #print('Voting Matrix:\n',pwcm.numpy())
        mat_confusion_P=confusion_matrix(patient_true, patient_pred)
        print('Confusion Matrix Patient:\n',mat_confusion_P)
        #patient_report = classification_report(patient_true, patient_pred, target_names=['Benign', 'Malignant'])
        #print(patient_report)
        score_precision_P = mat_confusion_P[1,1]/( mat_confusion_P[1,1] + mat_confusion_P[0,1] )*100
        acc_P = (mat_confusion_P[1,1]+mat_confusion_P[0,0])/( mat_confusion_P[1,1] + mat_confusion_P[0,1] + mat_confusion_P[1,0] + mat_confusion_P[0,0] )*100


        # REC = TP/(TP+FN)
        score_recall_P    = mat_confusion_P[1,1]/( mat_confusion_P[1,1] + mat_confusion_P[1,0] )*100
        # specificity = TN/(TN+FP)
        specificity_P   = mat_confusion_P[0,0]/( mat_confusion_P[0,0] + mat_confusion_P[0,1] )*100
        #NPV  = TN/(TN+FN)
        NPV_P = mat_confusion_P[0,0]/( mat_confusion_P[0,0] + mat_confusion_P[1,0] )*100
            
            # F1 = 2*PRE*( REC/(PRE+REC)
        score_f1_P = 2*score_precision_P*( score_recall_P/(score_precision_P+score_recall_P) )
        print( '\n..........Patient-wise Result......'  )
        print( 'Accuracy: %.3f' % acc_P )
        print( 'Precision or PPV: %.3f' % score_precision_P)
        print( 'NPV: %.3f' % NPV_P)
        print( 'Specificity %.3f' % specificity_P)
        print( 'Sensitivity or Recall: %.3f' % score_recall_P )
        print( 'F1: %.3f' % score_f1_P ) 
        kappa_P = cohen_kappa_score(patient_true, patient_pred)
        print('Cohens kappa: %f' % kappa_P)
        mc_P= matthews_corrcoef(patient_true, patient_pred)
        print('Correlation coeff: %f' % mc_P)


        # print( '\n.........Image-wise Result......'  )
        # mat_confusion=confusion_matrix(true, pred)
        #     #f1_score = f1_score(true,pred)
        # print('Confusion Matrix:\n',mat_confusion)
        #     #print('Precision: {},Recall: {}, Accuracy: {}'.format(precision*100,recall*100,accuracy*100))
        # score_precision = mat_confusion[1,1]/( mat_confusion[1,1] + mat_confusion[0,1] )*100
        # acc = (mat_confusion[1,1]+mat_confusion[0,0])/( mat_confusion[1,1] + mat_confusion[0,1] + mat_confusion[1,0] + mat_confusion[0,0] )*100


        # # REC = TP/(TP+FN)
        # score_recall    = mat_confusion[1,1]/( mat_confusion[1,1] + mat_confusion[1,0] )*100
        # # specificity = TN/(TN+FP)
        # specificity   = mat_confusion[0,0]/( mat_confusion[0,0] + mat_confusion[0,1] )*100
        # #NPV  = TN/(TN+FN)
        # NPV = mat_confusion[0,0]/( mat_confusion[0,0] + mat_confusion[1,0] )*100
            
        #     # F1 = 2*PRE*( REC/(PRE+REC)
        # score_f1 = 2*score_precision*( score_recall/(score_precision+score_recall) )
        # print( 'Accuracy: %.3f' % acc )
        # print( 'Precision or PPV: %.3f' % score_precision )
        # print( 'NPV: %.3f' % NPV )
        # print( 'Specificity %.3f' % specificity)
        # print( 'Sensitivity or Recall: %.3f' % score_recall )
        # print( 'F1: %.3f' % score_f1 ) 
        # kappa = cohen_kappa_score(true, pred)
        # print('Cohens kappa: %f' % kappa)
        # mc= matthews_corrcoef(true, pred)
        # print('Correlation coeff: %f' % mc)
        # #print(report)
        return (pred, true, soft)
feature_extract = True
sm = nn.Softmax(dim = 1)
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    ])

class TestDataset(Dataset) :
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
            patient_class = 1000
        elif data_path_BW.split("\\")[-3] == "Malignant" :
            label = 1
            patient_class = 2000
        else :
            raise Exception('invalid path')
        patient = data_path_BW.split("\\")[-2]
        patient_num = (int) (patient[ patient.find('(')+1 : patient.find(')') ]) + patient_class        
        return image, label, patient_num

    def __len__(self) : 
        length = len(self.BW)
        return length      

# test_data= datasets.ImageFolder(test_dir,transform=test_transforms)
test_data = TestDataset(test_dir1, test_dir2, transform=test_transforms)

num_workers = 0
print("Number of Samples in Test ",len(test_data))
test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
result = test(model, criterion)

# preds, true, soft = result
# images_path = test_loader.dataset.BW
# #images_path -> [ [images path, label] * 835 ]

# with open(f"majority_test1.csv", "w") as f:
#     wr = csv.writer(f)
#     wr.writerow(["file", "prob_0", "prob_1",  "pred", "label"])
#     for i in range(len(preds)):
#         f = os.path.basename(images_path[i])
#         #print(f)
#         prob_0 = round(soft[i][0], 6)
#         prob_1 = round(soft[i][1], 6)
#         pred = preds[i]
#         label = true[i]
#         wr.writerow([f, prob_0, prob_1, pred, label])
