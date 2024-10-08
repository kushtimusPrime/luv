## Installation
```
conda create --name unet_env
conda activate unet_env
python3 -m pip install torch==2.4 torchvision==0.19
python3 -m pip install lightning
cd ~/U-Net/fc-vision
python3 setup.py install
python3 -m pip install -e .

```
## Usage

Step 1: Setup needle data folder with prime_setup_needle_data_folder.py

Given as input, you have your data folder that should have

    |-- data

        |-- data_collection_1

            |-- data_scenario_1

                |-- images_left

                |-- left_masked

                |-- images_right

                |-- right_masked

            |-- data_scenario_2

            |-- ...
            
        |-- data_collection_2
        
        |-- ...

Output will look like

    |-- prime_data

        |-- images
        
        |-- targets

Step 2: Verify data is good. Use prime_data_verification.py. Use arrow keys to go through data and see the mask overlaid on the iamge. If it is bad, press 'd' to delete that image pair

Step 3: Split into training and testing data with prime_split_data.py

Step 4: To train the model, run train.py

Step 5: To figure out how good the model is/what you should set the masking threshold to, run prime_evaluate_iou.py and pick the threshold with the highest mean IoU.