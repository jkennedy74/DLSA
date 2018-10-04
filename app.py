import os
from flask import Flask, request, jsonify, render_template, redirect

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Load Larger LSTM network and generate text
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as k

app = Flask(__name__)

# empty list to capture filename to use for analysis
predicted_text = []
seed = []
sample_text = []
raw_sentiment = []
pred_sentiment = []

@app.route("/")
def index():

    """Return the homepage."""
    return render_template("index.html")

@app.route('/corpus', methods = ['POST'])
def corpus():

    option = request.form['optionsRadios']
    print("The corpus chosen is '" + option + "'")

    if option == 'option1':
        filepath = os.path.join('data','text', 'seuss.txt')
        modelname = os.path.join('data', 'weights', 'checkpoint-22-0.8293-seuss.hdf5')
    if option == 'option2':
        filepath = os.path.join('data','text', 'trump.txt')
        modelname = os.path.join('data', 'weights', 'checkpoint-10-1.6465-trump.hdf5')
    if option == 'option3':
        filepath = os.path.join('data','text', 'illiad.txt')
        modelname = os.path.join('data', 'weights', 'checkpoint-19-1.3924-illiad.hdf5')
    if option == 'option4':
        filepath = os.path.join('data','text', 'timemachine.txt')
        modelname = os.path.join('data', 'weights', 'checkpoint--49-1.4686-timemachine.hdf5')

    k.clear_session()

    # read the text file and covert to lowercase
    raw_text = open(filepath, encoding = "ISO-8859-1").read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(raw_text)
    n_vocab = len(chars)

    # prepare the dataset of input to output pairs encoded as integers
    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)

    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))

    # load the network weights
    filename = modelname
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # pick a random seed
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    for value in pattern:
        seed.append(int_to_char[value])

    for i in range(280):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        bob = prediction[0]
        index = numpy.random.choice(len(bob), p=bob)
        # index = numpy.argmax(prediction)
        result = int_to_char[index]
        predicted_text.append(result)
        # sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print ("\nDone.")

    
    raw_start = numpy.random.randint(0, n_chars-280)
    raw_end = raw_start + 280
    raw_sent = raw_text[raw_start:raw_end]
    pred_sent = ''.join(predicted_text)

    result1 = analyzer.polarity_scores(raw_sent)
    result2 = analyzer.polarity_scores(pred_sent)

    sample_text.append(raw_sent)
    raw_sentiment.append(result1)
    pred_sentiment.append(result2)

    return redirect('/')

@app.route("/sample_text")
def sample():

    return jsonify(sample_text)

@app.route("/lstm_output")
def rnn():

    return jsonify(''.join(predicted_text))

@app.route("/seed")
def rand_seed():

    return jsonify(''.join(seed))

@app.route("/raw_result")
def seeds():

    return jsonify(raw_sentiment)

@app.route("/pred_result")
def preds():

    return jsonify(pred_sentiment)

if __name__ == "__main__":
    app.run(debug=True)
