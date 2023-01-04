from flask import Flask, request, render_template
from cachelib  import SimpleCache
from modeling_bilstm import BiLSTM
from transformers import BertTokenizerFast
import torch

vocab_size = 3000
batch_size = 64
tokenizer = BertTokenizerFast(vocab_file = "tokenizer/vocab_size_{}/vocab.txt".format(str(vocab_size)), lowercase=False, strip_accents=False)

device = torch.device('cpu')
model = BiLSTM(len(tokenizer), 128, 2, 64, 1, 0.3)
model.load_state_dict(torch.load("model/vocab_size_3000_hidden_dim_128_embedding_dim_64_lstm_num_layers_1_parameter_size_146434/base_acc_8708_RunningTime_0-21-25/BiLSTM_model.pt", map_location=device))

def predict_sentiment(model, sentence, tokenizer):
    model.eval()
    tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokens["input_ids"]
    prediction = model(input_ids)
    softmax = torch.nn.Softmax(dim=1)
    return softmax(prediction).tolist()[0]

cache = SimpleCache()
app = Flask(__name__)

def find_cache(sentence):
    if cache.get(sentence) is None:
        label = predict_sentiment(model, sentence, tokenizer)
        cache.set(sentence, label, timeout=3000)
    return cache.get(sentence)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    sentence = request.form["sentence"]
    negative, positive = find_cache(sentence)
    output = "<h3>{}<br> {}%로 부정적인 문장 입니다! <br> {}%로 긍정적인 문장 입니다! </h3>".format(sentence, negative, positive)
    return output

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8000", debug=True)
