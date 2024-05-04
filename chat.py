import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize,stem,bagOfWords


#make sure the device supports CUDA
device = torch.device('cuda' if torch.cuda.is_available() else"cpu")


with open('intents.json','r') as f:
    intents = json.load(f)
FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data["allWords"]
tags= data["tags"]
modelState= data["modelState"]



model = NeuralNet(inputSize, hiddenSize,outputSize).to(device)
model.load_state_dict(modelState)
model.eval()

botName = "Shane"
print("Let's chat! type 'quit' to exit")
while True:
    sentences = input("You: ")
    if sentences == 'quit':
        break
    sentences = tokenize(sentences)
    X = bagOfWords(sentences,allWords)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # check the probablity is high enough
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() >0.75:
    # loop through all tags in intents and check if it fits
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{botName}: {random.choice(intent['responses'])}")
    else:
        print(f"{botName}: I do not understand...")