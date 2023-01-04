FROM pytorch/pytorch

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY serving/app.py app.py
COPY serving/download_model.py download_model.py
COPY serving/templates templates
COPY model model
COPY tokenizer tokenizer

ENTRYPOINT ["python", "app.py"]
