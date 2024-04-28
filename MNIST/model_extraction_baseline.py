import torch.optim as optim
import torchvision
from utils_mnist import *
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import torch.utils.data as Data

torch.manual_seed(10)

def train_steal_epoch(one_model,one_dataloader,one_optimizer):
    one_model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    for i, data in enumerate(one_dataloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        one_optimizer.zero_grad()
        outputs = one_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        one_optimizer.step()
        running_loss += loss.item()
    print(f'Loss: {running_loss / len(trainloader)}')

# load data
transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

from torch.utils.data import random_split
train_size = int(0.7 * len(trainset))
test_size = len(trainset) - train_size
new_train_dataset, new_steal_dataset = random_split(trainset, [train_size, test_size])
trainloader = DataLoader(new_train_dataset, batch_size=64, shuffle=True)
steal_loader = DataLoader(new_steal_dataset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=True)

# load victim model
victim_model = CNN()
victim_model.load_state_dict(torch.load('victim_mnist_model.pth'))
victim_model.cuda()
victim_model.eval()
# obtain stealing data
steal_data=None
steal_label=None
with torch.no_grad():
    for images, labels in steal_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = victim_model(images)
        if steal_data is not None:
            steal_data=torch.cat((steal_data,images),dim=0)
            steal_label = torch.cat((steal_label, outputs), dim=0)
        else:
            steal_data = images
            steal_label = outputs
steal_dataset = Data.TensorDataset(steal_data, steal_label)
meta_loader = DataLoader(steal_dataset, batch_size=128, shuffle=True)
# load stealing model
steal_model = CNN().cuda()
steal_optimizer = optim.SGD(steal_model.parameters(), lr=0.005)
# teain the stealing model
for k in range(20):
    train_steal_epoch(steal_model, meta_loader, steal_optimizer)
    calculate_test_accuracy(steal_model, testloader)

calculate_agreement_accuracy(victim_model,steal_model,testloader)
