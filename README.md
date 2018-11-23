# Siamese-RPN-pytorch
- [**Tensorflow version**](https://github.com/makalo/Siamese-RPN-tensorflow.git) has been available by my classmates[**makalo**](https://github.com/makalo).  
- This is a re-implementation for [**High Performance Visual Tracking with Siamese Region Proposal Network**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) with PyTorch, which is accepted at CVPR2018.  
- Please cite [**paper**](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) if you find this useful.  
- Another version [**zkisthebest/Siamese-RPN**](https://github.com/zkisthebest/Siamese-RPN) have **lots of bugs**, so I have to re-implement it. 
- Code_v1.0 is available for traning, you should change your dataset as **vot2013 format**.

## Getting Started
### Performance
<div align=center><img width="700" height="700" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/test2.gif"/></div>


### Network introduction  
<div align=center><img width="700" height="360" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/network.png"/></div>

### Environment  
- python=3.6  
- pytorch=0.4.0  
- cuda=9.0  
- shapely=1.6.4

## Downloading VOT2013 Dataset
```
wget http://data.votchallenge.net/vot2013/vot2013.zip 
```

## Downloading YouTube-bb Data
```
git clone https://github.com/mbuckler/youtube-bb.git

python3 download.py ./dataset 12
```

### Train phase 

```
git clone https://github.com/songdejia/siamese-RPN

cd code_v1.0

python train_siamrpn.py --dataroot=/PATH/TO/YOUR/DATASET --lr=0.001
```

### Visualization for debug

**bbox in detection**  
green -- ground truth which is got by pos anchor shift with reg_target  
red   -- bbox which is got by pos anchor with reg_pred  
black -- bbox with highest score

<div align=center><img width="400" height="400" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/bbox_in_detection.jpg"/></div>


**proposal in original image**
<div align=center><img width="640" height="360" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/bbox_in_origin.jpg"/></div>


## Authors

* **Bo Li** - *paper* - [Siamese-RPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)
* **De jiasong** - *code* - [Siamese-RPN-pytorch](https://github.com/songdejia/siamese-RPN)
* **Makalo**     - *code* - [Siamese-RPN-tensorflow](https://github.com/makalo/Siamese-RPN-tensorflow.git)











