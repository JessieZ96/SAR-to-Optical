# SAR-to-Optical

This is an open source composed of three parts: Classification, Restoration and Translation. If you use our code, please consider citing our paper. Thanks.

## Classification

* nn.py
 
To choose which model to train. To decide whether you should use pretrained or not.

* measure.py

To get the best results in the checkpoints and the confusion matrices.

* compare.ipynb

To compare the results of different models or different data. Meanwhile, draw the curves of loss and accuracy.

* confusion_matrix.ipynb

To draw the confusion matrices in the form of block diagram.

## Restoration

In this part, we were inspired by Keyan Ding et al.

* images

There are two kinds of images in it. One is distorted, the other is original. The goal is to restore the distorted image to the original one.

* examples

Run "recover.py" to implement the recovery process. 

For more details, go to https://github.com/dingkeyan93/IQA-optimization.

## Translation

In this part, Pix2pixHD is used as the backbone. The code borrows heavily from https://github.com/NVIDIA/pix2pixHD.

* datasets

The SAR-Optical remote sensing images are obtained by random sampling from the “SEN1-2 dataset” provided by Schmitt et al. We divided them into five categories of scene: Farmland, Forest, Gorge, River and Residential.

## Contact Us

If you have any questions, please contact us (jiexinz@seu.edu.cn, zjjee@nuaa.edu.cn).

For the experiment results, please see [Quality Assessment of SAR-to-Optical Image Translation](https://www.mdpi.com/2072-4292/12/21/3472) and [Feature-Guided SAR-to-Optical Image Translation](https://ieeexplore.ieee.org/document/9063491).





