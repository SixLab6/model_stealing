import torch
import torch.nn as nn

ngpu = 1
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def calculate_test_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy:',accuracy)

def calculate_agreement_accuracy(one_model,two_model,one_dataloader):
    one_model.eval()
    two_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in one_dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = one_model(images)
            ano_outputs=two_model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, ano_predicted = torch.max(ano_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == ano_predicted).sum().item()
    accuracy = 100 * correct / total
    print('Agreement:',accuracy)
