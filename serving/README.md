# bert-serving
TensorFlow Hub에 존재하는 distilkobert_sentence_encoder(문장 쌍의 유사도를 예측하는 Model)을 Flask로 Serving합니다. 불필요한 동작을 하지 않기 위해, 입력 문장에 대한 결과를 Caching합니다. Dockerfile을 제공하여 쉽게 이용할 수 있습니다. 정말 최소한의 기능만 사용해 본 Toy Project입니다.

## Use
```
docker build -t bert-serving ./bert-serving
docker run -it --rm -p 8000:8000 bert-serving
```

## Reference
- https://tfhub.dev/jeongukjae/distilkobert_sentence_encoder/1
- https://hidden-loca.tistory.com/29
- https://www.tensorflow.org/hub/tf2_saved_model?hl=ko
- https://wings2pc.tistory.com/entry/%EC%9B%B9-%EC%95%B1%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%94%8C%EB%9D%BC%EC%8A%A4%ED%81%ACPython-Flask-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC%ED%8F%B4%EB%8D%94-%EA%B5%AC%EC%84%B1
