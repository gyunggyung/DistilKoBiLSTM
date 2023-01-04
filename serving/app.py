from flask import Flask, request, render_template
from cachelib  import SimpleCache
from modeling_bilstm import BiLSTM
from transformers import BertTokenizerFast
import torch

vocab_size = 3000
batch_size = 64
tokenizer = BertTokenizerFast(vocab_file = "tokenizer/vocab_size_{}/vocab.txt".format(str(vocab_size)), lowercase=False, strip_accents=False)

model = BiLSTM(len(tokenizer), 128, 2, 64, 1, 0.3)
model.load_state_dict(torch.load("model/distil_new/vocab_size_3000_loss_rate_10_temperature_2/BiLSTM_model.pt"))
model.to("cuda")

def predict_sentiment(model, sentence, tokenizer):
    model.eval()
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokens["input_ids"].to("cuda")
    prediction = torch.argmax(model(input_ids))
    if prediction:
        return "긍정"
    return "부정"

cache = SimpleCache()
app = Flask(__name__)

def find_cache(key):
    if cache.get(key) is None:
        sentence = tf.constant([key])
        label = predict_sentiment(model, sentence, tokenizer)
        cache.set(key, label, timeout=3000)
    return cache.get(key)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    sentence = request.form["sentence"]
    return "<h3>해당 문장은 {}%문장 입니다!</h3".format(find_cache(sentence))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000", debug=True)
