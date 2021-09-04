# Relabeling-Crowd-Counting

This is the official Tensorflow implementation of the paper submission: Relabeling for Small-Data Crowd Count Estimation via Self Recalibration.

## Code

### Install Dependencies

The code is used with Python 3.6, requiring the packages listed below.

```
tensoflow==1.14.0
opencv-python
pillow
scipy
numpy
```
The packages can be easily installed by pip install.

### Train

1. Download the public ShanghaiTech raw dataset. [Google Drive Link](https://drive.google.com/file/d/1w1E6n7O70dw2iDZDqTi8x_Y0tCapQ_ka/view?usp=sharing)

2. Download the initial weight of the VGG backbone. [Google Drive Link](https://drive.google.com/file/d/1cnPsluk7tDZXnnjpObAD3cLXiWWipkBk/view?usp=sharing)

3. Unzip the downloaded file and modify the path of it to the same directory of this repository.

4. Run the python file for the data preprocessing.

  `python preprocess_ShanghaiTech.py --data-dir both_A_B`

5. Run the python file for the first training stage with relabeling.

  `python Self_Recalibration_First_Stage.py`

6. After the first training, run the python file for the scale re-estimation.

  `python Self_Recalibration_Re-estimate.py`

7. Run the python file for the second training stage.

  `python Self_Recalibration_Second_Stage.py`

8. We also provide the training of three baselines, please check by yourselves.

  `python Baseline_bayesianloss.py`  ...etc
