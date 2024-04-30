import torch
import torch.nn as nn
import torch.optim as optim
from utils_imdb import *
from torch.utils.data import TensorDataset,RandomSampler,DataLoader,SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hyper parameter
MAX_WORDS = 10000  # imdb’s vocab_size
MAX_LEN = 200      # max length
BATCH_SIZE = 256
EMB_SIZE = 128   # embedding size
HID_SIZE = 128   # lstm hidden size
DROPOUT = 0.2
# load train set
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
# TensorDataset
train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
# create a model
model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
optimizer = optim.Adam(model.parameters())

best_acc = 0.0
PATH = 'victim_imdb.pth'
for epoch in range(1, 21):
    train(model, DEVICE, train_loader, optimizer, epoch)
    acc = test(model, DEVICE, test_loader)
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), PATH)
    print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

best_model = Model(MAX_WORDS, EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
best_model.load_state_dict(torch.load(PATH))
test(best_model, DEVICE, test_loader)
