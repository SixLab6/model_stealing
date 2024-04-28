import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
from utils_mnist import *
import numpy as np
import torch.utils.data as Data
import torchvision
import torch.optim as optim

torch.manual_seed(123)
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

def predict_img(img):
    with torch.no_grad():
        prediction = victim_model(img)
    output = torch.nn.functional.softmax(prediction, dim=1)
    result=torch.argmax(output,dim=1)
    return result

def generate_nearSample(img,x,y):
    with torch.no_grad():
        feat = autoencoder.reverse(img)
        feat = feat + np.random.uniform(x, y, (1, 128, 4, 4))
        feat = feat.to(torch.float32)
        near_noise = autoencoder.reconstruction(feat)
    return near_noise

def generate_img(idx):
    seed=np.random.randint(0,100000)
    torch.manual_seed(seed)
    fixed_noise = torch.randn(1, nz, 1, 1).to(device)
    fixed_label = label_1hots[idx]
    fixed_label = torch.unsqueeze(fixed_label, dim=0)
    # Generate the Fake Images
    with torch.no_grad():
        fake = netG(fixed_noise.cpu(), fixed_label.cpu())
    return fake

def generate_label(target):
    fake_rondom_img = []
    fake_img = []
    number=300
    for i in range(number):
        fake_rondom_img.append(generate_img(target))
    for i in range(number):
        if predict_img(fake_rondom_img[i]) == target:
            fake_img.append(fake_rondom_img[i])
    return fake_img

def generate_near(fake_img,target):
    fake_samples=[]
    for i in range(len(fake_img)):
        fake_samples.append(fake_img[i][0])
        start, end = 0, 0.02
        for j in range(5):
            tep = generate_nearSample(fake_img[i], start, end)
            tep=(tep-0.5)*2
            if predict_img(tep)==target:
                fake_samples.append(tep[0])
            else:
                break
            start = start + 0.02
            end = end + 0.02
    return fake_samples

def get_new_label(onemodel,oneset):
    onemodel.eval()
    mylabel=None
    with torch.no_grad():
        for x in oneset:
            if mylabel is not None:
                mylabel=torch.cat((mylabel,onemodel(x)),dim=0)
            else:
                mylabel=onemodel(x)
    print(mylabel.shape)
    return mylabel
# Hyperparameter
dataroot = "data"
workers = 0
batch_size = 100
image_size = 28
nc = 1
num_classes = 10
nz = 100
ngf = 64
ndf = 64
num_epochs = 20
lr = 0.0002
beta1 = 0.5
ngpu = 1
# load some test samples of mnist
transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)
# Label one-hot for G
label_1hots = torch.zeros(10,10)
for i in range(10):
    label_1hots[i,i] = 1
label_1hots = label_1hots.view(10,10,1,1).to(device)
# load the trained large model
netG = Generator(0)
netG.load_state_dict(torch.load('last_model.pt'), strict=False)
netG.eval()
# load the trained victim model
victim_model = CNN()
victim_model.load_state_dict(torch.load('victim_mnist_model.pth'))
victim_model.eval()
# load the trained augmentation model
autoencoder = ComplexAutoencoder()
autoencoder.load_state_dict(torch.load('autoencoder_model.pth'))
autoencoder.eval()
# create new samples with the large model
new_data,new_label=[],[]
mnist_number=10
for i in range(mnist_number):
    lable_samples=generate_label(i)
    nearSamples=generate_near(lable_samples,i)
    tep_label = [i] * len(nearSamples)
    new_data = new_data + nearSamples
    new_label = new_label + tep_label
    print(len(new_data),len(new_label))
tep_label=torch.LongTensor(new_label)
new_data=torch.FloatTensor(torch.stack(new_data))
new_label=get_new_label(victim_model,new_data)
torch_dataset = Data.TensorDataset(new_data, new_label)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle=True
)
# train the reversed model
new_model = Reverse_Generator()
new_model.cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(new_model.parameters(), lr=0.05)
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs=inputs.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        outputs = new_model(labels)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(loader)}')
