# DistilKoBiLSTM
Transformer 이후 다양한 모델은 NLP Task에서 획기적인 성능을 보여줍니다. 동시에 무시무시한 Parameter Size와 Inference 속도를 보여줍니다. 이를 개선하기 위한 많은 방법이 나왔고, 해당 Repository에서는 LSTM model에 `Knowledge Distillation`를 진행합니다. [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136) 논문에서 영감을 얻었습니다.

[네이버 쇼핑, Steam 리뷰](https://github.com/bab2min/corpus/tree/master/sentiment), [네이버 영화 리뷰](https://github.com/e9t/nsmc) Dateset으로 한국어 감정분석 Binary Classification Task를 진행합니다. 해당 Dateset으로 FineTuning한 Transformer base Teacher Model로 BiLSTM를 Knowledge Distillation합니다. Distilling 과정에서 Teacher Model은 logits 값만 사용하기 때문에, 대부분의 모델(`BERT, ELECTRA, BART, GPT, XLNet, T5, ETC...`)을 사용할 수 있습니다.

Distilling 과정에서 epoch마다 Teacher Model output을 Inference 한다면, 많은 시간이 소요됩니다. 이를 방지하기 위해, 사전에 Dataset index 별 Teacher Model의 logits 값을 가진 dictionary를 만듭니다. 이를 이용하여, train 속도가 수십 ~ 수천 배 이상 빨라집니다.

## How to Use
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

### Hyperparameter tuning
`main.ipynb`에서 다양한 Hyperparameter를 설정할 수 있습니다. 아래 부분을 수정하여, 사용할 수 있습니다.

``` python
vocab_size = 3000
hidden_dim = 128
embedding_dim = 64
loss_rate = 0.1
temperature = 2
train_epoch = 30

teacher_path = "teacher_model/KoELECTRA-Small-v3/"

distil_trainer = Distil_Trainer(hidden_dim = hidden_dim, embedding_dim = embedding_dim, lstm_num_layers = 1, train_epoch = train_epoch,
                                out_put_dir = "distil_scheduler/vocab_size_{}_loss_rate_{}_temperature_{}/StepLR".format(str(vocab_size), str(int(loss_rate * 100)), temperature), tokenizer = tokenizer,
                                teacher_output = teacher_output, loss_rate = loss_rate, temperature = temperature)

```

### tokenizer
적은 Parameters를 가진 Student Model은, tokenizer vocab size에 따라서, Model Size가 크게 변합니다. 해당 Repository는 한국어 감정분석 Dataset으로 [다양한 vocab size(2000~9000)](https://github.com/gyunggyung/DistilKoBiLSTM/tree/main/tokenizer)를 만들었습니다. `Hugging Face BertWordPieceTokenizer`로 tokenizer를 만들었습니다.

BPE, SentencePiece, 형태소 분석기 등 다른 방식의 tokenizer를 만들거나, 다른 Dataset을 사용할 수 있습니다. tokenizer를 저장할 때는 `tokenizer/` directory에 `vocab_size_n` 형식으로 만드는 것을 추천합니다. 

tokenizer 종류에 따라서, `utils.py` `Line 21~23` 부분을 수정해야 할 수도 있습니다. tokenizer는 문자열로 구성된 list 형태의 sentences를 입력받아, tensor 형태로 반환합니다.

```python
        else:
            tokens = self.tokenizer(sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        X = tokens["input_ids"]
```

## Result

| Model                    | Total Parameters | Model Size |   Acc |
| ------------------------ | ---------------: | ---------: | ----: |
| `BERT-Large`             |        335174458 |      1.34G |     - |
| `BERT-Base-Multilingual` |        177974523 |       714M | 87.54 |
| `KoBERT`                 |         92186880 |       369M | 90.26 |
| `KoELECTRA-Base-v3`      |        112330752 |       452M | 90.98 |
| `KoELECTRA-Small-v3`     |         14056192 |      56.6M | 89.90 |
| `DistilKoBiLSTM-Base`    |           391170 |       1.5M | 88.20 |
| `DistilKoBiLSTM-Smail`   |           146434 |      547KB | 87.17 |

- 실험에 사용된 `DistilKoBiLSTM`는 `KoELECTRA-Small-v3`을 Teacher Model로 사용했습니다.
- `DistilKoBiLSTM-base` 기준 각 모델 별 Parameter Size 감축: `BERT-Large` 869배, `BERT-Base-Multilingual` 455배 ,`KoBERT` 235배, `KoELECTRA-Base-v3` 287배, `KoELECTRA-Small-v3` 36배. 엄청난 크기 차이 속에서, 최대 2.78%의 Acc 차이를 보입니다. 오히려 성능이 더 좋은 경우도 있습니다. `DistilKoBiLSTM-smail`의 경우 `DistilKoBiLSTM-base`보다 Acc가 1% 정도 떨어지지만, Parameter Size가 2.67배 더 작습니다.


### Hyperparameter and Acc and Train Time

| Model                    | vocab size | hidden dim | embedding dim | loss rate | temperature |   Acc | Step | Train Time |
| ------------------------ | ---------: | ---------: | ------------: | --------: | ----------: | ----: | ---: | ---------: |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           2 | 87.74 |   30 |   00:49:32 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |          10 | 87.74 |   30 |   00:49:26 |
| **`DistilKoBiLSTM-base`** |  **3000** |    **128** |        **64** | **0.1** | **1** | **88.20** | **30** | **00:50:29** |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |          10 | 87.73 |   30 |   00:49:27 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.5 |          10 | 87.92 |   30 |   00:48:51 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.9 |          10 | 87.61 |   30 |   00:49:02 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           1 | 86.83 |   30 |   00:44:41 |
| **`DistilKoBiLSTM-Smail`** | **3000** |     **64** |        **32** | **0.1** | **2** | **87.17** | **30** | **00:44:41** |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |          10 | 86.67 |   30 |   00:44:34 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |           1 | 86.76 |   30 |   00:44:40 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |          10 | 86.77 |   30 |   00:44:34 |



## Todo
- [ ] Add Relu
- [ ] Add Attention
- [ ] Clean model path
- [ ] Save BiLSTM Hyperparameter
- [ ] checkpoint Restore and continue training
- [ ] Use CPU
- [ ] Make class Simple Trainer
- [X] Write Acc
- [ ] Edit TensorBoard
- [ ] Make distil.py file
- [ ] Data Augmentation and Additional Distilling
- [ ] Web Serving Upload
- [ ] Edit get_teacher_output Function

### Distilling Teacher Model
- [ ] KoELECTRA-Base-v3(Need to learn again)
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
- [범용적인 감정 분석(극성 분석)은 가능할까](https://bab2min.tistory.com/657)