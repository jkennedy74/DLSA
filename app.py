import sys
import os
import io
import numpy as np
from flask import Flask, request, redirect, url_for, jsonify, render_template

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Uploads'

@app.route("/")
def index():
    """Return the homepage."""
    return render_template("index.html")

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     data = {"success": False}
#     if request.method == 'POST':
#         if request.files.get('file'):
#             # read the file
#             file = request.files['file']

#             # read the filename
#             filename = file.filename

#             # create a path to the uploads folder
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#             file.save(filepath)


if __name__ == "__main__":
    app.run(debug=True)
