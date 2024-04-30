import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import vgg16
import numpy as np
import torch.nn.functional as F
from own_encoder import AutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# generate some neighbor samples
def get_near_sample(img,target):
    tep=copy.deepcopy(img)
    tep=tep.reshape(1,3,32,32)
    tep=tep.to(device)
    target=target.to(device)
    with torch.no_grad():
        pre=vgg16(tep)
        nears_lab=pre
        nears = tep
        for i in range(20):
            rand_noise = np.random.uniform(0, 0.02, (1, 64, 8, 8))
            rand_noise=rand_noise*(i+1)
            rand_noise=torch.Tensor(rand_noise).to(device)
            feat=model.get_feature(tep)
            feat=feat+rand_noise
            reverse_img=model.reverse_img(feat)
            # plt.imshow(reverse_img[0].cpu().permute(1,2,0))
            # plt.show()
            reverse_img=data_transform(reverse_img)
            prediction=vgg16(reverse_img)
            if target==torch.argmax(prediction):
                nears=torch.cat([nears,reverse_img],dim=0)
                nears_lab=torch.cat([nears_lab,prediction],dim=0)
            else:
                break
    return nears,nears_lab

# load the trained autoencoder
model = AutoEncoder()
model=torch.load('encoder.pt')
model=model.to(device)
model.eval()
# load the trained victim model
vgg16=torch.load('vgg16.pt')
vgg16 = vgg16.to(device)
vgg16.eval()
# load the created samples
data_transform=transforms.Compose([transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
transform = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
train_dataset = datasets.ImageFolder(root='./data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_img=None
train_lab=None
for idx, (data,target) in enumerate(train_loader):
    if train_img is not None:
        train_img=torch.cat([train_img,data],dim=0)
        train_lab=torch.cat([train_lab,target],dim=0)
    else:
        train_img = data
        train_lab = target
# select available samples
datas,labels=[],[]
cnt=0
for i in range(len(train_img)):
    data,label=train_img[i].reshape(1,3,32,32),train_lab[i]
    data, label= data.to(device),label.to(device)
    with torch.no_grad():
        pre=vgg16(data)
        output=torch.argmax(pre)
        if output==label and torch.max(F.softmax(pre))>0.7:
            datas.append(data)
            labels.append(label)
# expand available samples
new_data,new_label=None,None
for i in range(len(datas)):
    tep,tep_lab=get_near_sample(datas[i],labels[i])
    if new_data is not None:
        new_data = torch.cat([new_data, tep], dim=0)
        new_label = torch.cat([new_label, tep_lab], dim=0)
    else:
        new_data = tep
        new_label = tep_lab
# save all created samples
torch.save(new_data,'data.pt')
torch.save(new_label,'label.pt')
