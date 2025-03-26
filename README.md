# MI-DPRC


## Introduction
This is the code for [***Active learning for Object Detection with Vectorized Dual Pseudo Loss and Multiple Instance Offset Constraint***]

### Task Description

In this paper, we propose an active learning method for object detection with vectorized dual pseudo loss and multiple instance offset constraint to select the most informative images for detector training.



### Innovation

(1) We propose a method for evaluating regression loss based on the offset discrepancy loss vector. This method addresses the issue of the inability to acquire dual regression labels in object detection by utilizing the discrepancy between enhanced and original base box vectors, combined with the regressor parameter information constraint.


(2) We propose a regression entropy and anchor offset guided adaptive weighting method. This approach includes uncertainty quantification of regression information and a weighted branch based on anchor offset to assess the regression/classification information weighting coefficient, effectively mitigating the impact of high information noisy instances.

(3) We propose a novel Active Learning Method MI-DPRC. It synergistically integrates the information quality with diversity-aware instance selection. The distance entropy modulates mul-instances tradeoffs and instance-level diversity sampling optimizes data efficiency by eliminating redundant patterns in high-information-density regions.

## Getting Started


### Data Preparation

Please download VOC2007 datasets ( *trainval* + *test* ) and VOC2012 datasets ( *trainval* ) from:

VOC2007 ( *trainval* ): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

VOC2007 ( *test* ): http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

VOC2012 ( *trainval* ): http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

And after that, please ensure the file directory tree is as below:

```
MI-DPRC
|
`-- data
    -- VOCdevkit
        |
        |--VOC2007
        |  |
        |  |--ImageSets
        |  |--JPEGImages
        |  `--Annotations
        `--VOC2012
           |--ImageSets
           |--JPEGImages
           `--Annotations
```
For convenience, we use COCO style annotation for Pascal VOC active learning. Please download [trainval_0712.json](https://drive.google.com/file/d/1GIAmjGbg47dZFJjGYf2p-dU1z4V7pACQ/view?usp=sharing).

### Train and Test
We recommend you to use a GPU but not a CPU to train and test, because it will greatly shorten the time.

```shell
python tools/train.py 
python tools/test.py 
```

### Run active learning
- You can run active learning using a single command with a config file. For example, you can run Pascal VOC RetinaNet experiments by
```shell
python tools/AL_mian.py --config al_configs/MI-DPRC.py --model retinanet
```