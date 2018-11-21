# Siamese-RPN-pytorch
[Tensorflow version](https://github.com/makalo/Siamese-RPN-tensorflow.git) has been available by my classmates.

This is a re-implementation for [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) with PyTorch, which is accepted at CVPR2018.
Former work [zkisthebest](https://github.com/zkisthebest/Siamese-RPN) cant backward properly, so some modification has been done.
Code_v1.0 is available for traning, you should change your dataset as vot2013 format.


## Getting Started
You need to prepare for dataset with the format like [vot2013](http://www.votchallenge.net/vot2013/)


### Prerequisites

```
pip install shapely
```


## Running the training 

```
git clone https://github.com/songdejia/siamese-RPN

cd code_v1.0

python train_siamrpn.py --dataroot=/PATH/TO/YOUR/DATASET --lr=0.01
```

### Visualization for debug

** bbox in detection ** 

green -- ground truth which is got by pos anchor shift with reg_target

red   -- bbox which is got by pos anchor with reg_pred

black -- bbox with highest score

<div align=center><img width="400" height="400" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/bbox_in_detection.jpg"/></div>


** proposal in original image **
<div align=center><img width="640" height="360" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/bbox_in_origin.jpg"/></div>


## Authors

* **Bo Li** - *paper* - [Siamese-RPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)
* **De jiasong** - *code* - [Siamese-RPN-pytorch](https://github.com/songdejia/siamese-RPN)
* **Makalo**     - *code* - [Siamese-RPN-tensorflow](https://github.com/makalo/Siamese-RPN-tensorflow.git)











