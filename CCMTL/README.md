# Directories:
- Data_Preparation -- Matlab Scripts for Ground Truth and Data processing.
- Final Models -- Models Saved After Trained.
- Output -- Information saved in txt of Loss Function and Counts.
- src - Main files and Modules [crowd_count.py | data_loader.py | evaluate_model.py | models.py | network.py | timer.py | utils.py ]
******
## How to Run this Code:

1. Download ShanghaiTech Dataset

2. Create Directory

``` mkdir ROOT/data/original/shanghaitech/ ``` 

3. Save "part_A_final" under ROOT/data/original/shanghaitech/

4. Save "part_B_final" under ROOT/data/original/shanghaitech/

5. cd ROOT/data_preparation/

6. run create_gt_test_set_shtech.m in matlab to create ground truth files for test data

7. cd ROOT/data_preparation/

    ```run create_training_set_shtech.m in matlab to create training and validataion set along with ground truth files```
## Test & Train
1. For Testing the models.
``` python test.py``` 

2. For Training the Network.
``` python train.py```
