# [CVPR19] DeepCO<sup>3</sup>: Deep Instance Co-segmentation by Co-peak Search and Co-saliency (Oral paper)
### Authors: [Kuang-Jui Hsu](https://www.citi.sinica.edu.tw/pages/kjhsu/), [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_en.html), [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)

## Abstract
In this paper, we address a new task called instance cosegmentation. Given a set of images jointly covering object instances of a specific category, instance co-segmentation aims to identify all of these instances and segment each of them, i.e. generating one mask for each instance. This task is important since instance-level segmentation is preferable for humans and many vision applications. It is also challenging because no pixel-wise annotated training data are available and the number of instances in each image is unknown. We solve this task by dividing it into two sub-tasks, co-peak search and instance mask segmentation. In the former sub-task, we develop a CNN-based network to detect the co-peaks as well as co-saliency maps for a pair of images. A co-peak has two endpoints, one in each image, that are local maxima in the response maps and similar to each other. Thereby, the two endpoints are potentially covered by a pair of instances of the same category. In the latter subtask, we design a ranking function that takes the detected co-peaks and co-saliency maps as inputs and can select the object proposals to produce the final results. Our method for instance co-segmentation and its variant for object colocalization are evaluated on four datasets, and achieve favorable performance against the state-of-the-art methods.


<img src="https://github.com/KuangJuiHsu/DeepCO3/blob/master/Images/CVPR19.PNG" height="400"/>

+ PDF: [High-Resolution](http://cvlab.citi.sinica.edu.tw/images/paper/cvpr-hsu19.pdf), [Low-Resolution](http://cvlab.citi.sinica.edu.tw/images/paper/cvpr-hsu19-lowres.pdf)
+ Supplementary material: [High-Resolution](https://drive.google.com/file/d/1zNB1oydDUMQGLbZie1rJgvHTPjmDnYTC/view?usp=sharing), [Low-Resolution](https://drive.google.com/file/d/1aYR88gVmZHedZUK43M49MqZVWQ4z3A8F/view?usp=sharing)

<p>Please cite our paper if this code is useful for your research.</p>
<pre><code>
@inproceedings{HsuCVPR19,
  author = {Kuang-Jui Hsu and Yen-Yu Lin and Yung-Yu Chuan},
  booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {DeepCO$^3$: Deep Instance Co-segmentation by Co-peak Search and Co-saliency Detection},
  year = {2019}
}
</code></pre>

---

# Codes for DeepCO<sup>3</sup>

- Contact: [Kuang-Jui Hsu](https://www.citi.sinica.edu.tw/pages/kjhsu/)
- Last update: 2019/04/04
- Platform: Ubuntu 14.04, [MatConvnet 1.0-beta24](http://www.vlfeat.org/matconvnet/) (Don't support any installation problem of MatConvnet.)


## Demo for all stages: "RunDeepInstCoseg.m"
- Including the MatConvnet and the corresponding "mex" files
- May be slightly different from the ones in paper because of the randdom seeds.

## Datasets (about 34 GB):
- Including the images, ground-truth masks, salinecy maps and object proposals
- [GoogleDrive](https://drive.google.com/file/d/1IDyC8NXQdOZEaji6GKQZbh9uZ5B2r_79/view?usp=sharing)

## Results reported in the papers (about 4 GB):
- Only including the DeepCO<sup>3</sup> results 
- [GoogleDrive](https://drive.google.com/file/d/1sMr11hbmc6w3GZAOKy5pbEZxyHBJtb8z/view?usp=sharing)

## Download Codes from GoogleDrive :
- [GoogleDrive](https://drive.google.com/file/d/1NnEVkrrrYyi5oNRKuIlQ6dkdupC5kHbB/view?usp=sharing)
