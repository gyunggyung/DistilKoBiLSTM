### Hyperparameter and Acc and Train Time

- optimizer는 `Adam` `lr = 0.001, betas=(0.9, 0.999), eps=1e-08`으로 `PyTorch` 기본 값을 사용했습니다. 
- 나열하지 않은 scheduler의 값은 위 optimizer의 값을 따릅니다. `StepLR`는 `gamma = 0.9`를 사용했습니다. `CyclicLR`는 `base_lr = 0.0005, step_size_up = len_train_iter // 2, cycle_momentum = False`를 사용했습니다.
- 이외에 다른 설정은 아래를 참고해주세요.
- `DistilKoBiLSTM-Tiny` model의 경우 distilling을 하지 않은 `KoBiLSTM-Tiny` model과 성능이 비슷하지만, 이는 model이 매우 작아서 발생하는 문제로 보입니다. `DistilKoBiLSTM-Tiny`은 17 step에서 Acc가 85를 넘었습니다. 반면 `KoBiLSTM-Tiny`은 step 63에 이를 달성했습니다.

## Use Teacher Model KoELECTRA-Small-v3

| Model                    | vocab size | hidden dim | embedding dim | loss rate | temperature |   Acc | Step | Train Time | step_size |
| ------------------------ | ---------: | ---------: | ------------: | --------: | ----------: | ----: | ---: | ---------: | --------: |
| `DistilKoBiLSTM-Large`   |       3000 |        256 |           128 |       0.1 |           2 | 89.12 |  100 |   04:11:10 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           1 | 87.84 |   30 |   00:49:15 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           2 | 87.74 |   30 |   00:49:32 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |           3 | 87.83 |   30 |   00:46:07 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |         0 |          10 | 87.74 |   30 |   00:49:26 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           1 | 88.20 |   30 |   00:50:29 |         5 |
| **`DistilKoBiLSTM-base`** |  **3000** |    **128** |       **64** | **0.1** | **2** | **89.12** | **100** | **02:49:21** |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 87.98 |   30 |   00:46:27 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           3 | 88.09 |   30 |   00:46:19 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           4 | 87.94 |   30 |   00:46:19 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |          10 | 87.76 |   30 |   00:46:25 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.5 |          10 | 87.92 |   30 |   00:48:51 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.9 |          10 | 87.61 |   30 |   00:49:02 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           1 | 86.83 |   30 |   00:44:41 |         5 |
| **`DistilKoBiLSTM-Smail`** | **3000** |     **64** |        **32** | **0.1** | **2** | **87.17** | **30** | **00:44:41** |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           2 | 87.16 |   43 |          - |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           2 | 88.07 |  100 |   02:42:16 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           2 | 88.33 |  100 |   02:41:31 |        10 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           3 | 86.91 |   30 |   00:45:07 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           4 | 87.02 |   30 |   00:44:47 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |          10 | 86.67 |   30 |   00:44:34 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |           1 | 86.76 |   30 |   00:44:40 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.9 |          10 | 86.77 |   30 |   00:44:34 |         5 |
| **`DistilKoBiLSTM-Mini`** |  **3000** |     **26** |      **8** |   **0.1** | **2** | **85.17** | **100** | **02:13:56** |         5 | 
| **`DistilKoBiLSTM-Tiny`** |  **3000** |     **16** |        **16** |   **0.1** |      **2** | **86.17** | **59** | **-** |         5 |

### Use CyclicLR scheduler
| Model                    | vocab size | hidden dim | embedding dim | loss rate | temperature |   Acc | Step | Train Time | step_size |
| ------------------------ | ---------: | ---------: | ------------: | --------: | ----------: | ----: | ---: | ---------: | --------: |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 88.84 |  100 |   02:47:51 |         5 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           2 | 88.22 |   55 |          - |         5 |


## Use Teacher Model KoELECTRA-Base-v3/

| Model                    | vocab size | hidden dim | embedding dim | loss rate | temperature |   Acc | Step | Train Time | step_size |
| ------------------------ | ---------: | ---------: | ------------: | --------: | ----------: | ----: | ---: | ---------: | --------: |
| `DistilKoBiLSTM-Large`   |       3000 |        256 |           128 |       0.1 |           2 | 89.27 |   55 |          - |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 88.45 |  100 |   02:51:58 |         5 |
| `DistilKoBiLSTM-base`    |       3000 |        128 |            64 |       0.1 |           2 | 88.56 |  100 |   02:55:33 |        10 |
| `DistilKoBiLSTM-Smail`   |       3000 |         64 |            32 |       0.1 |           2 | 88.33 |  100 |   02:41:31 |        10 |
