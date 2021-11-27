import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

loaded_model = load_model("NLP_embdedding_model.h5")

num_words = 15000
max_tokens = 61
tokenizer = joblib.load("tokenizer.pckl")


def f(r):
    tokens = tokenizer.texts_to_sequences([r])
    tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
    res = loaded_model.predict(tokens_pad)[0][0]
    if res <.5:
        return res,"negative sentiment"
    else:
        return res,"positive sentiment"

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/",methods=["GET", "POST"])
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        text = request.form['sentiment']
        res = f(text)
        return render_template("predict.html", result=res)

if __name__ == "__main__":
    app.run()

