import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as Data
from utils_cifar import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the victim model
victim_model=torch.load('vgg16.pt').cuda()
victim_model.eval()
# create a new model
new_vgg=get_vggmodel().to(device)
# load a test set
transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
# load the created samples
new_data=torch.load('data.pt')
new_label=torch.load('label.pt')
new_dataset=Data.TensorDataset(new_data,new_label)
new_loader = DataLoader(new_dataset, batch_size=64, shuffle=True)
# train the stealing model
criterion = nn.MSELoss()
optimizer = optim.SGD(new_vgg.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(new_vgg.parameters(), lr=0.001)
for epoch in range(50):
    running_loss = 0.0
    new_vgg.train()
    for i, data in enumerate(new_loader, 0):
        inputs, labels = data
        # inputs=train_transform(inputs)
        inputs, labels=inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = new_vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(new_loader)}")
    accuracy = test_accuracy(new_vgg, testloader)
    print(f"Accuracy of the network on the test images: {accuracy}%")
