import torch
import torch.nn as nn
from torchvision.models import vgg19

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vggmodel():
    vgg16 = vgg19(pretrained=True)
    for param in vgg16.parameters():
        param.requires_grad = True
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, 2)
    return vgg16

def get_testacc(model,val_loader,criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {correct / total}")
