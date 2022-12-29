### Hyperparameter and Acc and Train Time

| Model                    | vocab size | hidden dim | embedding dim | loss rate | temperature |   Acc | Step | Train Time |
| ------------------------ | ---------: | ---------: | ------------: | --------: | ----------: | ----: | ---: | ---------: |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           1 | 87.84 |   30 |   00:49:15 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           2 | 87.74 |   30 |   00:49:32 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           3 | 87.83 |   30 |   00:46:07 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |          10 | 87.74 |   30 |   00:49:26 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           1 | 88.20 |   30 |   00:50:29 |
| **`DistilKoBiLSTM-base`** |  **3000** |    **128** |       **64** | **0.1** | **2** | **89.12** | **100** | **02:49:21** |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 87.98 |   30 |   00:46:27 |
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
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |          10 | 86.67 |   30 |   00:44:34 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |           1 | 86.76 |   30 |   00:44:40 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |          10 | 86.77 |   30 |   00:44:34 |
| **`DistilKoBiLSTM-Mini`** |  **3000** |     **16** |        **16** |   **0.1** |      **2** | **86.17** | **59** | **-** |

- optimizer는 `Adam` `lr = 0.001, betas=(0.9, 0.999), eps=1e-08`으로 `PyTorch` 기본 값을 사용했습니다. scheduler는 `StepLR` `step_size = 1, gamma = 0.9`를 사용했습니다. Step 30을 넘는 값들은 `step_size = 5`로 만들었습니다.