### Siamese-RPN

This is a re-implementation for [High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) with PyTorch, which is accepted at CVPR2018.


This work is based on [zkisthebest](https://github.com/zkisthebest/Siamese-RPN) but I think some tricks is available.

### Introduction
+ 1.Network Architecture
<div align=left><img width="900" height="300" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/network.png"/></div>

+ 2.Visualization for dataloader(template 127 * 127 and detection 271 * 271)
<div align=left><img width="200" height="200" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/000_a_template.jpg"/></div>
<div align=left><img width="400" height="400" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/001_detection_input.jpg"/></div>
<div align=left><img width="200" height="200" src="https://github.com/songdejia/siamese-RPN/blob/master/screenshot/001_detection_output.jpg"/></div>

+ 3.How CNN backpropogation works in out1 * out2(convolution).
	It is easy to fall into the trap of abstracting away the learning process — believing that you can simply stack arbitrary layers together and backprop will “magically make them work” on your data.
	Chinese people [strongly recommend](https://www.zhihu.com/question/27239198) step1:compute tmp d, step2:compute dw
	(if you want to change something, compute d toward it)
	Prerequirement [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) [chinese](https://zh.wikipedia.org/wiki/%E9%9B%85%E5%8F%AF%E6%AF%94%E7%9F%A9%E9%98%B5)









