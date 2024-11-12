# End-to-End Monocular Visual Odometry via Local Map Feature Association with Transformers
## Dataset
Download the [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)  
   
The data structure should be as follows:
  
    |---data
        |---00
            |---image_2
                |---000000.png
                |---000001.png
                |---...
            |---image_3
            |---00.txt
            |---calib.txt
            |---times.txt
        |---01
        |---...
    |---poses
        |---00.txt
        |---01.txt
        |---...


## Setup
Create a virtual environment using Anaconda and activate it:
  
    conda env create -f requirement.yml
    conda activate vo

## Usage
### Training
run:

    python train.py --device cuda --weighted_loss 1.0 --seq_len 31 --epoch 200
  
### Inference and  evalutaion
You can predict the trajectory of a sequence by revising *predict.py* and running:

    python predict.py

## Result
