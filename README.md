# EgoFlow: detect if a person is looking at me with RGB and corresponding optical-flow sequences.
The second price in [2022 Ego4D Looking-at-Me Challenge](https://eval.ai/web/challenges/challenge-page/1624/overview)

### Introduction
This is a method proposed for "2022 Ego4D Looking-at-Me Challenge", exploiting [Ego4D dataset](https://ego4d-data.org/docs/challenge/#dataset) and an optical flow dataset, this dataset is created with the RGB images in Ego4D by a neural network called [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official).

### Installation
```
conda create -n egoflow python=3.8 
conda activate egoflow
conda install pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0
pip install -r requirements.txt
```

### Model
1. FlowFormer
Create a folder
```
mkdir checkpoints
```
and put [sintel.pth](https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_) in this folder.
2. A pre-trained EgoFlow [model](https://drive.google.com/file/d/1YT8yd_fsC0cBTxDX8cYw5docpWrZxRM1/view?usp=share_link) for testing

### Create Optical Flow Dataset
Run all "*_odd.py" and "*_even.py" files to generate the optical flows. Please check the roots of datasets in these python files.

### Train
```
python run.py --model GazeLSTM --exp_path output_train --num_workers 16 --batch_size 64 --gamma 0.5
```

### Test
```
python run.py --eval --checkpoint output_train/checkpoint/best.pth --model GazeLSTM --exp_path output_test --num_workers 16 --batch_size 128
```
