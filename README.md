# Computational_Image_Aesthetic

## Table of Contents

* [AVA dataset](#ava-dataset)
* [Baseline Network](#baseline-network)
* [Aesthetic Network](#aesthetic-network)

## AVA dataset

<img src="./img/2.png" alt="2" style="zoom: 67%;" />

### AVA.txt

<img src="./img/3.png" alt="3"  />

Total: about 250000 images

### tags.txt

<img src="./img/4.png" alt="4" style="zoom:67%;" />

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

<img src="./img/1.png" alt="1" style="zoom: 50%;" />

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
   
   - `data_check.py`
2. Split data set -> train set & test set & validation set
   
   - `data_create.py`
   
### Training log

1. baseline network. Then save weights.
   - dataset-inputs: (?, 224, 224, 3); dataset-outputs: (?, 2)
   - `train_baseline.py`