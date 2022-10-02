
import json
import torch, nltk, numpy as np
from nltk.stem.snowball import SnowballStemmer
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('pattern_matches.json', 'r') as f:
    pattern_matches = json.load(f)
subtopics = [value for ele in pattern_matches for value in ele.values() if type(value) == type('str')]

FILE = "infiseriesmodel.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

init_string = 'Welcome to InfiSeries. It can (unfortunately only attempt to) categorise questions inputted :)\nA project for Deep Learning Week 2022.'
print('#' * 64)
print(init_string)
print('#' * 64)

print("\nSend the questions! (type 'quit' to exit)")

def bag_of_words(tokenized_sentence, words):

    # stem each word
    sentence_words = [SnowballStemmer('english').stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag

while True:
    # sentence = "do you use credit cards?"
    sentence = input("\nQuestion: ")
    if sentence == "quit":
        break

    sentence = nltk.word_tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for subtopic in subtopics:
            if tag == subtopic:
                print(f'Seems like the topic is {tag}')

    else:
        print(f"Not getting it :(")