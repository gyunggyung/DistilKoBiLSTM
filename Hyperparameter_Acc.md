### Hyperparameter and Acc and Train Time

| Model                    | vocab size | hidden dim | embedding dim | loss rate | temperature |   Acc | Step | Train Time |
| ------------------------ | ---------: | ---------: | ------------: | --------: | ----------: | ----: | ---: | ---------: |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           1 | 87.84 |   30 |   00:49:15 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           2 | 87.74 |   30 |   00:49:32 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           3 | 87.83 |   30 |   00:46:07 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |          10 | 87.74 |   30 |   00:49:26 |
| **`DistilKoBiLSTM-base`** |  **3000** |    **128** |        **64** | **0.1** | **1** | **88.20** | **30** | **00:50:29** |
| **`DistilKoBiLSTM-base`** |  **3000** |    **128** |       **64** | **0.1** | **2** | **89.12** | **100** | **02:49:21** |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 87.98 |   30 |   00:46:27 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 89.12 |   30 |   step 100 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           3 | 88.09 |   30 |   00:46:19 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           4 | 87.94 |   30 |   00:46:19 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |          10 | 87.76 |   30 |   00:46:25 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.5 |          10 | 87.92 |   30 |   00:48:51 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.9 |          10 | 87.61 |   30 |   00:49:02 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           1 | 86.83 |   30 |   00:44:41 |
| **`DistilKoBiLSTM-Smail`** | **3000** |     **64** |        **32** | **0.1** | **2** | **87.17** | **30** | **00:44:41** |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           2 | 87.16 |   43 |          - |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           3 | 86.91 |   30 |   00:45:07 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           4 | 87.02 |   30 |   00:44:47 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           5 | 87.16 |   30 |    step 43 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |          10 | 86.67 |   30 |   00:44:34 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |           1 | 86.76 |   30 |   00:44:40 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |          10 | 86.77 |   30 |   00:44:34 |
| **`DistilKoBiLSTM-Mini`** |  **3000** |     **16** |        **16** |   **0.1** |      **2** | **86.17** | **59** | **-** |



- optimizer는 `Adam` `lr = 0.001, betas=(0.9, 0.999), eps=1e-08`으로 `PyTorch` 기본 값을 사용했습니다. scheduler는 `StepLR` `step_size = 1, gamma = 0.9`를 사용했습니다. Step 30을 넘는 값들은 `step_size = 5`로 만들었습니다. 모든 LSTM Model은 30 epoch 동안 학습을 진행했습니다.


step_size 5로 같은 조건이네 다행이다!
BiLSTMmodel_hidden_dim_128_embedding_dim_64_step100_lstm_num_layers_1_parameter_size_391170_acc_8912


biLSTMmodel_hidden_dim_16_embedding_dim_16_step60_lstm_num_layers_1_parameter_size_52418_acc_8615


BiLSTMmodel_hidden_dim_64_embedding_dim_32_step43_lstm_num_layers_1_parameter_size_146434_acc_8716


base?

EndModel_biLSTMmodel_hidden_dim_16_embedding_dim_8_step100_lstm_num_layers_1_parameter_size_27394_acc_8509_RunningTime_0-33-51
109kb

EndModel_biLSTMmodel_hidden_dim_32_embedding_dim_8_step100_lstm_num_layers_1_parameter_size_34882_acc_8547_RunningTime_0-34-52
139kb

EndModel_biLSTMmodel_hidden_dim_64_embedding_dim_32_step100_lstm_num_layers_1_parameter_size_146434_acc_8751_RunningTime_0-48-51



EndModel_biLSTMmodel_hidden_dim_128_embedding_dim_64_step100_lstm_num_layers_1_parameter_size_391170_acc_8779_RunningTime_1-12-18

biLSTMmodel_hidden_dim_128_embedding_dim_64_step100_lstm_num_layers_1_parameter_size_391170_acc_8840