FROM python:3.6.8

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY serving serving
COPY model model
COPY tokenizer tokenizer
COPY modeling_bilstm.py serving/modeling_bilstm.py

ENTRYPOINT ["python", "serving/app.py"]