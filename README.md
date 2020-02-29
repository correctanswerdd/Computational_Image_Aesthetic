# Computational_Image_Aesthetic

## Table of Contents

[TOC]

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

x: (batch_size, 224, 224, 3)

y: (batch_size, 132), in which 132 = 66 * 2. 66 is the number of tags.

Split data set:

- 0.96 train set
- 0.005 test set
- 0.005 validation set

### Architecture

<img src="./img/1.png" alt="1" style="zoom: 50%;" />

### Training log

#### Step1

Train A+B+output, and save weights.

#### Step 2

Read pre-trained-A, then train A+C+output

### Result 

output(A+C+output) loss ~ 0.7