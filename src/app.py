# from flask import Flask, render_template, request
# from model import predict_sentiment

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     review = request.form['review']
#     sentiment = predict_sentiment(review)
#     return render_template('result.html', sentiment=sentiment, review=review)

# if __name__ == '__main__':
#     app.run(debug=True)











from flask import Flask, render_template, request
from model import predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = predict_sentiment(review)
    return render_template('result.html', sentiment=sentiment, review=review)

if __name__ == '__main__':
    app.run(debug=True)
