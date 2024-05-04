import json
from nltk_utils import tokenize,stem,bagOfWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)
allWords = []
tags = []
expectInputNOutput =[]
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent["patterns"]:
        word = tokenize(pattern)
        # tokenize returns an array
        allWords.extend(word)
        expectInputNOutput.append((word, tag))
ignoreWords = ['?','!','.',',']
allWords = [stem(word) for word in allWords if word not in ignoreWords]
# use sorted instead of .sort() cuz set allow us to get unique words and sorted() returns a sorted list
allWords = sorted(set(allWords))
tags = sorted(set(tags))

inputTrain=[]
outputTrain=[]
for (patternSetence,tag) in expectInputNOutput:
    bag = bagOfWords(patternSetence,allWords)
    inputTrain.append(bag)

    # the label should be the index of the tag in the tags array
    label = tags.index(tag)
    outputTrain.append(label) #CrossEntropyLoss

inputTrain = np.array(inputTrain)
outputTrain = np.array(outputTrain)

class ChatDataset(Dataset):
    def __init__(self):
        self.nSamples = len(inputTrain)
        self.xdata = inputTrain
        self.ydata = outputTrain
    def __getitem__(self, index):
        return self.xdata[index], self.ydata[index]
    def __len__(self):
        return self.nSamples
    
#Hyperparameters
batchSize = 8

hiddenSize = 8
# Different tags that we have
outputSize = len(tags)
# Suppose to be the size/length of the bag of words, which will be the same size of the all words array
inputSize = len(allWords)
learningRate = 0.001
numEpochs = 1000

dataset = ChatDataset()
trainLoader = DataLoader(dataset=dataset, batch_size=batchSize,shuffle=True, num_workers=0)

#make sure the device supports CUDA
device = torch.device('cuda' if torch.cuda.is_available() else"cpu")

model = NeuralNet(inputSize, hiddenSize,outputSize).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
# Actual trainning loop
for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        output = model(words)
        loss = criterion(output, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
#     if (epoch +1 )%100 ==0:
#         print(f'epoch{epoch+1}/{numEpochs}, loss={loss.item():.4f}')
# print(f'final loss, loss={loss.item():.4f}')

# Save the data
data = {
    "modelState" : model.state_dict(),
    "inputSize": inputSize,
    "outputSize": outputSize,
    "hiddenSize": hiddenSize,
    "allWords": allWords,
    "tags": tags,
}

#.pth stands for pythorch
FILE = "data.pth"
torch.save(data,FILE)
print(f"training complete, file save to {FILE}")
