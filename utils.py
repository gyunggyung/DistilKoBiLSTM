import time
import datetime
import torch
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, tokenizer, test_size = 0.1, valid_size = 0.05, tokenizer_type = "KoBertTokenizer", batch_size = 256):
        self.tokenizer = tokenizer
        self.test_size = test_size
        self.valid_size = valid_size
        self.tokenizer_type = tokenizer_type
        self.batch_size = batch_size

    def make_dataloder(self, data, data_type = "lstm"):
        sentences = list(data["document"])
        if data_type == "lstm":
            tokens = self.tokenizer.batch_encode_plus(sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        else:
            tokens = self.tokenizer(sentences, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        X = tokens["input_ids"]
        Y = list(data["label"])
        Y = torch.tensor(Y)
        idx = list(data["idx"])
        idx = torch.tensor(idx)
        dataset = TensorDataset(X, Y, idx)
        return DataLoader(dataset, batch_size = self.batch_size, pin_memory = True)

    def load_data(self, df):
        print("Start load data!!")
        start = time.time()
        train_data, test_data = train_test_split(df, test_size = self.test_size, random_state = 1)
        train_data, valid_data = train_test_split(train_data, test_size = self.valid_size, random_state = 1)
        train_iter = self.make_dataloder(train_data)
        test_iter  = self.make_dataloder(test_data)
        valid_iter = self.make_dataloder(valid_data)
        end = time.time()
        sec = (end - start)
        result = datetime.timedelta(seconds = sec)
        print("load data time : {}".format(result))
        return train_iter, test_iter, valid_iter

import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def make_tokenizer(start = 2000, end = 10000, step = 1000, data_file = 'dataset.txt'):
    for vocab_size in range(start, end, step):
        tokenizer = BertWordPieceTokenizer(lowercase=False)
        if vocab_size > 2000:
            limit_alphabet = 1500
        else:
            limit_alphabet = 1000
        min_frequency = 5
        print(vocab_size)
        tokenizer.train(files = data_file,
                        vocab_size = vocab_size,
                        limit_alphabet = limit_alphabet,
                        min_frequency = min_frequency)

        vocab_path = 'tokenizer/vocab_size_{}/'.format(vocab_size)

        createFolder(vocab_path)
        tokenizer.save_model(vocab_path)


import pickle

def input_list_positive_or_negative(sentences, model, tokenizer):
    tokens = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
    sequence_output = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])
    return sequence_output[0].cpu().tolist()

def make_teacher_output(df, teacher_path, tokenizer, Classification, batch_size = 32):
    checkpoint = teacher_path + "pytorch_model.bin"
    config_file = teacher_path + "config.json"
    teacher_model = Classification.from_pretrained(checkpoint, config = config_file).to("cuda")

    document_list = list(df["document"])
    idx_list = list(df["idx"])

    teacher_output = {}
    n = len(document_list)
    for i in range(0, n, batch_size):
        output_list = input_list_positive_or_negative(document_list[i:i+batch_size], teacher_model, tokenizer)
        now_idx = idx_list[i:i+batch_size]
        for output, idx in zip(output_list, now_idx):
            teacher_output[idx] = output
        if i / 32 % 100 == 0:
            print(i, i / n)

    return teacher_output

def get_teacher_output(teacher_path, tokenizer = None, Classification = None, df = None, batch_size = None):
    teacher_output_path = teacher_path + "teacher_output.pickle"
    if os.path.isfile(teacher_output_path):
        with open(teacher_output_path, 'rb') as fr:
            teacher_output = pickle.load(fr)
    else:
        teacher_output = make_teacher_output(df, teacher_path, tokenizer, Classification, batch_size)
    return teacher_output


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params