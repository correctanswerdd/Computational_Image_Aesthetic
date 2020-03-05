# Computational_Image_Aesthetic

## Table of Contents

* [AVA dataset](#ava-dataset)
* [Baseline Network](#baseline-network)
* [Aesthetic Network](#aesthetic-network)

## AVA dataset

<p align="center">
    <img src="./img/2.png" alt="Sample"  width="400">
  	<p align="center">
    	<em style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;"><span id = "img1">数据集文件结构</span></em>
</p>

### AVA.txt

<p align="center">
    <img src="./img/3.png" alt="Sample"  width="700">
  	<p align="center">
    	<em style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;"><span id = "img1">AVA.txt文件存储格式</span></em>
</p>

Total: about 250000 images

### tags.txt

<p align="center">
    <img src="./img/4.png" alt="Sample"  width="500">
  	<p align="center">
    	<em style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;"><span id = "img1">tags.txt文件存储格式/span></em>
</p>

tag_id + tag_description

Total: 66 tags

### Related codes

`data.py`

## Baseline Network

### Data set

#### Tag - Classification Task

x: (batch_size, 224, 224, 3)

y: (batch_size, 132), in which 132 = 66 * 2. 66 is the number of tags.

Split data set 'tag':

- 0.96 train set

- 0.005 test set

- 0.005 validation set

#### Score - Regression Task

x: (batch_size, 224, 224, 3)

y: (batch_size, 1)

Split data set 'score':

- 0.96 train set
- 0.005 test set
- 0.005 validation set

### Architecture

<p align="center">
    <img src="./img/1.png" alt="Sample"  width="700">
  	<p align="center">
    	<em style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;"><span id = "img1">baseline network</span></em>
</p>

### Related codes

`network.py` 

### Training log

#### Step1

Train A+B+output, and save weights.

#### Step 2

Read pre-trained-A, then train A+C+output

### Result 

AVA dataset binary classification accuracy ~ 0.3

## Aesthetic Network

### Data set

1. Check data set: remove lines with empty url in `AVA.txt` & create new url file `AVA_check.txt`
   
   - `check_data.py`
2. Split data set -> train set & test set & validation set
   
   - `create_data.py`
   
### Training log

1. baseline network. Then save weights.
   - dataset-inputs: (?, 224, 224, 3); dataset-outputs: (?, 2)
   - `train_baseline.py`