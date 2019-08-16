from flask import Flask, jsonify
from flask import render_template, request

from predict import my_predict

app = Flask(__name__)

@app.route('/', methods=['POST'])
def form_post():

    activate = request.form.get('activate')
    url = request.form['text']

    return render_template('index.html',
                            url=url,
                            activate=activate,
                            prediction=my_predict(url, visualise=False),
                            my_html=my_predict(url, visualise=True)
                            )
@app.route('/')
def landing_page():
    return render_template('index.html')


@app.route('/api/<path:url>', methods=['GET'])
def api_page(url):
    
    prediction = my_predict(url)

    return jsonify(dict(prediction))

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
