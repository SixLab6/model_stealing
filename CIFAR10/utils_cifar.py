import torch
import torch.nn as nn
from torchvision.models import vgg19

def test_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        for data in testloader:
            images, labels = data
            images, labels=images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_vggmodel():
    vgg16 = vgg19(pretrained=True)
    for param in vgg16.parameters():
        param.requires_grad = True
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, 10)
