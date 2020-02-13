import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request


def open_file(file):
    with open(file, 'r') as f:
        data = json.load(f)
        return data

data = open_file('data.json')
contractions = {
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that had",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they had",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what  will",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you had",
    "you'll": "you will",
    "you're": "you are",
}

lemma = WordNetLemmatizer()


def tokenize_and_stem_text(text):  # stem each word in text
    return [lemma.lemmatize(word.lower()) for word in text]


# get a list of all categories to train for: loves_me, loves_me_not
binary_categories = list(data.keys())
training_words = []
# a list of tuples with words in the sentence and category name
json_data = []


def read_training_data(data):  # modify training_words
    for category in data.keys():
        for text in data[category]:
            for word in text.split():
                if word.lower() in contractions:
                    text = text.replace(word, contractions[word.lower()])
            # remove  punctuation from the sentence
            text = re.sub("[^a-zA-Z' ]+", ' ', text)
            # extract words from each sentence and append to the word list
            training_words.extend(word_tokenize(text))
            json_data.append((word_tokenize(text), category))
    return json_data


# stem and lower each word and remove duplicates
tokenize_and_stem_text(training_words)
read_training_data(data)
# create training data
training = []
for item in json_data:
    # initialize our bag of words(bow) for each document in the list
    bag_vector = []
    # list of tokenized words for the pattern
    token_words = item[0]
    # stem each word
    token_words = [lemma.lemmatize(word.lower()) for word in token_words]
    print(f'{token_words}')
    # create our bag of words array
    for w in training_words:
        if w in token_words:
            bag_vector.append(1)
        else:
            bag_vector.append(0)
    out_row = list([0] * len(binary_categories))  # empty array for our output
    out_row[binary_categories.index(item[1])] = 1
    # our training set will contain a the bag of words model and the output row that tells
    # which category that bow belongs to.
    training.append([bag_vector, out_row])

# shuffle features,  turn into np.array bc tf takes in numpy array
# random.shuffle(training)
training = np.array(training)
print(f'training {training}')
# data contains the bag of words and labels contains the label/category
data = list(training[:, 0])
print(f'data {data}')
labels = list(training[:, 1])
print(f'labels {labels}')
# reset underlying graph data, clear defined variables and operations of the previous cell
tf.reset_default_graph()
# Build neural network: 3 layers. input_data layer is used for inputting (aka. feeding) data to a network. the input to your network has shape len(data[0])
print(f'len(data[0] {len(data[0])}, {data[0]}')
# none = unknown dimension, so can change total # samples processed in batch
net = tflearn.input_data(shape=[None, len(data[0])])
net = tflearn.fully_connected(net, 32)  # 32 hidden units/neurons
# softmax vs sigmoid:
net = tflearn.fully_connected(net, len(labels[0]), activation='softmax')
net = tflearn.regression(net)
# Define model and setup tensorboard: DNN automatically performs NN classifier tasks like training, prediction
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm). epochs = # times the network will see all data
model.fit(data, labels, n_epoch=1000, batch_size=16, show_metric=True)
# method takes in a sentence + list of all words, returns data in form that  can be fed to tensorflow

def clean_for_tf(text):
    input_words = tokenize_and_stem_text(word_tokenize(text))
    bag_vector = [0]*len(training_words)
    for input_word in input_words:
        for ind, word in enumerate(training_words):
            if word == input_word:
                bag_vector[ind] = 1
    return(np.array(bag_vector))


app = Flask(__name__)
@app.route("/sms", methods=['POST'])
def sms():
    resp = MessagingResponse()
    inbMsg = request.values.get('Body').lower().strip()
    # position of largest value: prediction
    tensor = model.predict([clean_for_tf(inbMsg)])
    resp.message(
        f'The message {inbMsg!r} corresponds to {binary_categories[np.argmax(tensor)]!r}.')
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)
