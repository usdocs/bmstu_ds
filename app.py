import flask
from flask import render_template
import numpy as np
from tensorflow import keras

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    if flask.request.method == 'POST':
        loaded_model = keras.models.load_model('models/nn_model')

        val1 = float(flask.request.form['val1'])
        val2 = float(flask.request.form['val2'])
        val3 = float(flask.request.form['val3'])
        val4 = float(flask.request.form['val4'])
        val5 = float(flask.request.form['val5'])
        val6 = float(flask.request.form['val6'])
        val7 = float(flask.request.form['val7'])
        val8 = float(flask.request.form['val8'])
        val9 = float(flask.request.form['val9'])
        val10 = float(flask.request.form['val10'])
        val11 = float(flask.request.form['val11'])
        val12 = float(flask.request.form['val12'])

        values = np.array([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]])

        y_pred = loaded_model.predict(values)
        return render_template('main.html', result=y_pred)

app.run()
