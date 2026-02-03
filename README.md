# Dense Haze Removal Based on Dynamic Collaborative Inference Learning for Remote Sensing Images
remote sensing, hazy dataset

Dense Haze Removal Based on Dynamic Collaborative Inference Learning for Remote Sensing Images

![Dense Haze Removal Based on Dynamic Collaborative Inference Learning for Remote Sensing Images](./main/fig2.tif)

The paper can be seen in the [Paper](https://ieeexplore.ieee.org/document/9895281)

# Dataset
## Dataset Structure

This project utilizes two primary datasets for training and testing:

### 1. Light Hazy Image Dataset (LHID)
Inspired by the method of synthesizing hazy image pairs, we synthesize the light hazy datasets. In the synthesis process:
- **Airlight $A$**: Randomly sampled in $[0.7, 1]$.
- **Transmission $t$**: Randomly sampled in $[0.35, 0.65]$.
- **Content**: 30,517 Remote Sensing Images (RSIs) selected from Google Earth for the training dataset.
- **Resolution**: 0.2–153 m.
- **Image Size**: $512 \times 512$.

### 2. Dense Hazy Image Dataset (DHID)
For dense hazy scenarios, we extract 500 different transmission maps from real hazy RSIs:
- **Clear Images**: Cropped clear images ($512 \times 512$) from the DLR 3k Munich Vehicle Aerial Image Dataset (MVAID), which contains 55 vehicle aerial images ($5616 \times 3744$) with a ground sampling distance of approximately 0.13 m.
- **Synthesis**: Extracted transmission maps and random airlight $A \in [0.7, 1]$ are added to create 14,990 dense hazy images.
- **Split**: 14,490 for training and 500 for testing.

Link: https://pan.baidu.com/s/13aW-khZZcLF3_1ax4H8GXQ?pwd=QW67 
Password: QW67

## Pre-trained Model
### 1. Download the pre-trained LHID/DHID/RICE/RSID model, put in ./checkpoints

   百度云盘: https://pan.baidu.com/s/17fsVMdB-VTcgfAeyGq_5DA   Password: HJ49
   GoogleDriver: https://drive.google.com/drive/folders/15SR3IYx8jT6ymLKx9XLU4eJp_X38ugOY?usp=drive_link
   
This repository contains the official implementation of DCIL.

## Environment Requirements
- Python 3.x
- PyTorch
- torchvision
- tqdm
- visdom (for logging)
- PIL
- numpy
- thop (for parameters and MACs calculation)

## Training

### 1. Configuration
Open `config.py` to set up your training environment:

- **Dataset Paths**: Modify `train_data_root` and `val_data_root` to point to your training and validation datasets.
  ```python
  train_data_root = '/path/to/your/train_dataset'
  val_data_root = '/path/to/your/val_dataset'
  ```
- **Pre-trained Model**: If you want to resume training or use a pre-trained weights, set `load_model_path`:
  ```python
  load_model_path = 'checkpoints/your_model.pth' # Set to None if training from scratch
  ```
- **Hyperparameters**: Adjust `batch_size`, `lr` (learning rate), and `max_epoch` as needed.

### 2. Start Training
Run the training script:
```bash
python Train.py
```
Make sure `visdom` is running if you want to monitor progress:
```bash
python -m visdom.server
```

## Testing

### 1. Data and Model Setup
Open `test.py` and modify the following paths:

- **Input Images**: Set the path to the directory containing hazy images:
  ```python
  # Line 17
  imgs = glob.glob('/path/to/your/hazy/images/*')
  ```
- **Output Directory**: Set where the dehazed results should be saved:
  ```python
  # Line 28
  output_path = '/path/to/save/results/'
  ```
- **Model Weight**: The testing script uses `load_model_path` from `config.py`. Ensure it points to the correct `.pth` file.

### 2. Run Inference
```bash
python test.py
```

## Dataset Structure
The `DehazingSet` expects a standard dehazing dataset structure, typically with `hazy` and `clear` subdirectories or paired files (depending on your `DehazingSet.py` implementation). Ensure your paths in `config.py` point to the root of these subfolders.

---
*Note: This code has been refactored for clarity and includes English comments and tqdm progress bars for better usability.*

# Citations
If our dataset and code are helpful to you, please cite:

@article{zhang2022dense,
  title={Dense haze removal based on dynamic collaborative inference learning for remote sensing images},
  author={Zhang, Libao and Wang, Shan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}



