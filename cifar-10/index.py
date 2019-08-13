from flask import Flask
from flask import render_template, request

from predict import my_predict

app = Flask(__name__)

@app.route('/', methods=['POST'])
def form_post():

    activate = request.form.get('activate')
    url = request.form['text']

    return render_template('index.html',
                            url=url,
                            prediction=my_predict(url, visualise=False),
                            my_html=my_predict(url, visualise=True)
                            )
@app.route('/')
def landing_page():
    return render_template('index.html')

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=6969, debug=True)
