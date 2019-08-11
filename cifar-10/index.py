from flask import Flask
from flask import render_template, request

from predict import predict

app = Flask(__name__)

@app.route('/', methods=['POST'])
def form_post():
    url = request.form['text']
    prediction = predict(url)
    return render_template('index.html', url=url, prediction=prediction)

@app.route('/')
def landing_page():
    return render_template('index.html')

if __name__ == '__main__':

    app.run(port=6969, debug=True)
