from flask import Flask, request
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
from twilio.twiml.messaging_response import MessagingResponse
import model

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
