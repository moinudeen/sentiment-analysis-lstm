from flask import Flask, render_template,request
import numpy as np
import keras.models
import re
from keras.preprocessing import text, sequence

import sys 
import os
sys.path.append(os.path.abspath("./model"))
from load import * 

app = Flask(__name__)

global model, graph
model, graph = init()

def preprocessText(rawtext):
	inp = text.text_to_word_sequence(rawtext)
	idx2w = np.load('idx2w.npy')
	w2idx = np.load('w2idx.npy').item()

	numeric_inp = [[ w2idx[x] if x in w2idx else w2idx[UNK] for x in inp]]
	numeric_inp = sequence.pad_sequences(numeric_inp,maxlen=maxlen)
	return numeric_inp


@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
	rawtext = request.args.get('inp')
	inp = preprocessText(rawtext)
	print (inp)
	with graph.as_default():
		out = model.predict(x)
		print(out)
		print(np.argmax(out,axis=1))
		print "debug3"
		response = np.array_str(np.argmax(out,axis=1))
		return response	
	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)
	#app.run(debug=True)
