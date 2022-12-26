# DistilKoBiLSTM
Transformer 이후 다양한 모델은 NLP Task에서 획기적인 성능을 보여줍니다. 동시에 무시무시한 Parameter Size와 Inference 속도를 보여줍니다. 이를 개선하기 위한 많은 방법이 나왔고, 해당 Repository에서는 LSTM model에 Knowledge Distillation를 진행합니다. [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136) 논문에서 영감을 얻었습니다.

[네이버 쇼핑, Steam 리뷰](https://github.com/bab2min/corpus/tree/master/sentiment), [네이버 영화 리뷰](https://github.com/e9t/nsmc) Dateset으로 한국어 감정분석 Binary Classification Task를 진행합니다. 해당 Dateset으로 FineTuning한 Transformer base Teacher Model로 BiLSTM를 Knowledge Distillation합니다. Distilling 과정에서 Teacher Model은 logits 값만 사용하기 때문에, 대부분의 모델(BERT, ELECTRA, BART, GPT, XLNet, T5, ETC...)을 사용할 수 있습니다.

Distilling 과정에서 epoch마다 Teacher Model output을 Inference 한다면, 많은 시간이 소요됩니다. 이를 방지하기 위해, 사전에 Dataset index 별 Teacher Model의 logits 값을 가진 dictionary를 만듭니다. 이를 이용하여, train 속도가 수십 ~ 수천 배 이상 빨라집니다.

## Use
```
git clone https://github.com/gyunggyung/DistilKoBiLSTM.git
```
위 명령어로 해당 Repository를 clone한 후, [Google Drive](https://drive.google.com/drive/my-drive)에 넣습니다. `main.ipynb` 파일을 Colab으로 실행합니다. Colab은 장시간 사용하지 않을 경우 Runtime이 끊어질 수도 있습니다. 이를 대비하기 위해, 개발자 모드(F12) Console에서 아래 코드를 붙여 넣는 것을 추천합니다.

### Runtime Disconnection Prevention
``` javascript
function ClickConnect() {
    // 백엔드를 할당하지 못했습니다.
    // GPU이(가) 있는 백엔드를 사용할 수 없습니다. 가속기가 없는 런타임을 사용하시겠습니까?
    // 취소 버튼을 찾아서 클릭
    var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); 
    buttons.forEach(function(btn) {
        btn.click();
    });
    console.log("1분 마다 다시 연결");
    document.querySelector("#top-toolbar > colab-connect-button").click();
}
setInterval(ClickConnect,1000*60);

```

## tokenizer
적은 Parameters를 가진 Student Model은, tokenizer vocab size에 따라서, Model Size가 크게 변합니다. 해당 Repository는 한국어 감정분석 Dataset으로 [다양한 vocab size(2000~9000)](https://github.com/gyunggyung/DistilKoBiLSTM/tree/main/tokenizer)를 만들었습니다. Hugging Face BertWordPieceTokenizer로 tokenizer를 만들었습니다.

BPE, SentencePiece, 형태소 분석기 등 다른 방식의 tokenizer를 만들거나, 다른 Dataset을 사용할 수 있습니다. `tokenizer/` directory에 `vocab_size_n` 형식으로 만드는 것을 추천합니다. 

tokenizer 종류에 따라서, `utils.py` `Line 21~23` 부분을 수정해야 할 수 있습니다. tokenizer는 문자열로 구성된 list 형태의 sentences를 입력받아, tensor 형태로 반환합니다.

```python
        else:
            tokens = self.tokenizer(sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        X = tokens["input_ids"]
```

## Todo
- [ ] Add Relu
- [ ] Add Attention
- [ ] Clean model path
- [ ] Save BiLSTM Hyperparameter
- [ ] Use CPU
- [ ] Make class Simple Trainer
- [ ] Write Acc
- [ ] Edit TensorBoard
- [ ] Make distil.py file
- [ ] Web Serving
- [ ] Edit get_teacher_output Function

### Distilling Teacher Model
- [ ] KoBERT
- [ ] DistilKoBERT
- [ ] KLUE-RoBERTa
- [ ] KoBART
- [ ] KoBigBird
- [ ] KoGPT2
- [ ] XLNet
- [ ] T5

## Reference

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
- [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108)
- [네이버 쇼핑, Steam 리뷰 말뭉치](https://github.com/bab2min/corpus/tree/master/sentiment)
- [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
- [CUSTOM DATASET으로 BERTTOKENIZER 학습하기](https://cryptosalamander.tistory.com/139)
- [tacchinotacchi/distil-bilstm](https://github.com/tacchinotacchi/distil-bilstm)
- [pvgladkov/knowledge-distillation](https://github.com/pvgladkov/knowledge-distillation)
- [monologg/KoELECTRA](https://github.com/monologg/KoELECTRA)
- [monologg/DistilKoBERT](https://github.com/monologg/DistilKoBERT)
- [HyejinWon/pytorch-nsmc-classification](https://github.com/HyejinWon/pytorch-nsmc-classification)
- [Google Colab 런타임 연결 끊김 방지](https://bryan7.tistory.com/1077)