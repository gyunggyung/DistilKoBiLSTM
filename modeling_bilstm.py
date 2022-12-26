import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embedding_dim, lstm_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = 0)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers = lstm_layers, 
                           bidirectional = True, 
                           batch_first = True, 
                           dropout = dropout) 
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.rnn(embed)
        #if self.lstm_num_layers >= 2:
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = self.linear(hidden)
        return hidden