import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modeling_bilstm import BiLSTM
from torch.utils.tensorboard import SummaryWriter

class Distil_Trainer():
    def __init__(self, input_dim = 8002, hidden_dim = 128, embedding_dim = 64, lstm_num_layers = 1, dropout = 0.3, tokenizer = None,
                 out_put_dir = "base", teacher_output = None, train_epoch = 5, lr = 0.001, step_size = 1, gamma = 0.9, 
                 scheduler_tpye = "StepLR", loss_rate = 0.5, temperature = 10, loss_option = "kl_div", len_train_iter = None):
        if tokenizer:
            self.tokenizer = tokenizer
            input_dim = len(tokenizer)
        self.model = BiLSTM(input_dim, hidden_dim, 2, embedding_dim, lstm_num_layers, dropout).to("cuda") # input_dim = len(tokenitan)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm_num_layers = lstm_num_layers
        self.dropout = dropout
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr) # default lr = 0.001
        if scheduler_tpye == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = step_size, gamma = gamma) # step_size = 2
        elif scheduler_tpye == "CyclicLR":
            self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr = lr / 25, max_lr = lr, 
                                                         step_size_up = len_train_iter // 2, cycle_momentum = False)
        self.criterion = nn.CrossEntropyLoss().to("cuda")
        self.out_put_dir = out_put_dir
        self.teacher_output = teacher_output
        self.loss_rate = loss_rate
        self.temperature = temperature
        self.train_epoch = train_epoch
        self.tb_suffix = "{}_input_{}_hidden_{}_embedding_{}_loss_rate_{}".format("_".join(out_put_dir.split("/")), input_dim, hidden_dim, embedding_dim, int(loss_rate * 100))
        self.tb_writer = SummaryWriter(log_dir = "/content/gdrive/MyDrive/DistilkoBiLSTM/logs", filename_suffix = self.tb_suffix) #(log_dir = "/content/gdrive/MyDrive/DistilkoBiLSTM/logs", filename_suffix = self.tb_suffix)
        self.loss_option = loss_option

    def __distil_loss(self, output, teacher_prob, real_label):
        alpha = self.loss_rate
        criterion_ce = nn.CrossEntropyLoss().to("cuda")
        if self.loss_option == "kl_div":
            criterion_kld = nn.KLDivLoss(reduction='batchmean').to("cuda")
            distillation_loss = criterion_kld(
                F.log_softmax(output / self.temperature, dim = 1), 
                F.softmax(teacher_prob / self.temperature, dim = 1)) * (self.temperature * self.temperature)
            return alpha * criterion_ce(output, real_label) + (1 - alpha) * distillation_loss
        elif self.loss_option == "mse":
            criterion_mse = nn.MSELoss().to("cuda")
            return alpha * criterion_ce(output, real_label) + (1 - alpha) * criterion_mse(output, teacher_prob)
        else:
            return criterion_ce(output, real_label)

    @staticmethod
    def __binary_accuracy(prediction, target):
        rounded_preds =  prediction.argmax(dim = 1)
        correct = (rounded_preds == target).float()
        return correct.sum() / len(correct)

    @staticmethod
    def __epoch_time(epoch_start):
        epoch_end = time.time()
        epoch_sec = (epoch_end - epoch_start)
        epoch_result = datetime.timedelta(seconds = epoch_sec)
        epoch_start = time.time()
        return epoch_result, epoch_start


    def train(self, train_iter):
        self.model.train()
        epoch_loss, epoch_acc = 0, 0
        epoch_start = time.time()        
        print("run iter : ", len(train_iter))
        for epoch, batch in enumerate(train_iter):
            if epoch % 100 == 1:
                print(" step: {} \n loss: {} \n acc: {}".format(epoch, loss, acc))
                self.tb_writer.flush()
                epoch_result, epoch_start = self.__epoch_time(epoch_start)
                print("epoch{} runing time : {}".format(epoch, epoch_result))

            self.optimizer.zero_grad()
            x, y, idx = batch
            x, y = x.to("cuda"), y.to("cuda")
            y_prob = self.model(x).squeeze(1)

            teacher_prob = [self.teacher_output[i.item()] for i in idx]
            teacher_prob = torch.tensor(teacher_prob).to("cuda")
            
            loss = self.__distil_loss(y_prob, teacher_prob, y)
            acc = self.__binary_accuracy(y_prob, y)

            loss.backward()
            self.optimizer.step()

            # self.tb_writer.add_scalar('loss'.format(self.tb_suffix), loss, epoch)
            # self.tb_writer.add_scalar('val_acc'.format(self.tb_suffix), acc, epoch)
            self.tb_writer.add_scalar('{}/loss'.format(self.tb_suffix), loss, epoch)
            self.tb_writer.add_scalar('{}/val_acc'.format(self.tb_suffix), acc, epoch)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(train_iter), epoch_acc / len(train_iter)

    def evaluate(self, valid_iter, epoch):
        self.model.eval()
        with torch.no_grad():
            eval_loss, eval_acc = 0, 0
            for batch in valid_iter:
                x, y, idx = batch
                x = x.to("cuda")
                y = y.to("cuda")

                y_prob = self.model(x).squeeze(1)
                teacher_prob = [self.teacher_output[i.item()] for i in idx]
                teacher_prob = torch.tensor(teacher_prob).to("cuda")
                
                loss = self.__distil_loss(y_prob, teacher_prob, y)
                acc = self.__binary_accuracy(y_prob, y)

                eval_loss += loss.item()
                eval_acc += acc.item()

            # self.tb_writer.add_scalar('ap_score', eval_acc / len(valid_iter), global_step = epoch)
            # self.tb_writer.add_scalar('ap_simple_loss', eval_loss / len(valid_iter), global_step = epoch)
            self.tb_writer.add_scalar('{}/ap_score'.format(self.tb_suffix), eval_acc / len(valid_iter), global_step = epoch)
            self.tb_writer.add_scalar('{}/ap_simple_loss'.format(self.tb_suffix), eval_loss / len(valid_iter), global_step = epoch)

        return eval_loss / len(valid_iter), eval_acc / len(valid_iter)

    @staticmethod
    def __create_folder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    def trainer(self, train_iter, valid_iter, test_iter, return_model = False):
        start = time.time()
        dir_path = "model/" + self.out_put_dir
        self.__create_folder(dir_path)
        epoch_start = time.time()
        for epoch in range(1, self.train_epoch + 1): # 5 epoch
            print("hidden_dim : {} embedding_dim : {}".format(self.hidden_dim, self.embedding_dim))
            train_loss, train_acc = self.train(train_iter)
            valid_loss, valid_acc = self.evaluate(valid_iter, epoch)
            print("[Epoch: %d] train loss : %5.3f | train accuracy : %5.3f" % (epoch, train_loss, train_acc))
            print("[Epoch: %d] val loss : %5.3f | val accuracy : %5.3f" % (epoch, valid_loss, valid_acc))

            self.scheduler.step() #lr scheduler

            parameter_size = sum(p.numel() for p in self.model.parameters())
            model_name = '/BiLSTMmodel_hidden_dim_{}_embedding_dim_{}_step{}_lstm_num_layers_{}_parameter_size_{}_acc_{}.pt'.format(self.hidden_dim, self.embedding_dim, epoch, self.lstm_num_layers, parameter_size, int(valid_acc * 10000))
            torch.save(self.model.state_dict(), dir_path + model_name)

            epoch_result, epoch_start = self.__epoch_time(epoch_start)
            print("epoch{} runing time : {}".format(epoch, epoch_result))

        end = time.time()
        sec = (end - start)
        result = datetime.timedelta(seconds = sec)
        print("runing time : {}".format(result))

        test_loss, test_acc = self.evaluate(test_iter, epoch)
        print('Test Loss: %5.2f | Test Acc: %5.2f '%(test_loss, test_acc * 100))

        result = str(result).split(".")[0].replace(":", "-")
        model_name = '/EndModel_BiLSTMmodel_hidden_dim_{}_embedding_dim_{}_step{}_lstm_num_layers_{}_parameter_size_{}_acc_{}_RunningTime_{}.pt'.format(self.hidden_dim, self.embedding_dim, epoch, self.lstm_num_layers, parameter_size, int(test_acc * 10000), result)
        torch.save(self.model.state_dict(), dir_path + model_name)
        
        if return_model:
            return self.model
        return None

    def predict_sentiment(self, sentence):
        self.model.eval()
        tokens = self.tokenizer(sentence, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
        input_ids = tokens["input_ids"].to("cuda")
        prediction = self.model(input_ids)
        return prediction