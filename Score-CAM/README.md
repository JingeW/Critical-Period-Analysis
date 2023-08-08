## Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks

We develop a novel post-hoc visual explanation method called Score-CAM, which is the first gradient-free CAM-based visualization method that achieves better visual performance (**state-of-the-art**). 

<img src="https://github.com/haofanwang/Score-CAM/blob/master/pics/comparison.png" width="100%" height="100%">

Paper: [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Wang_Score-CAM_Score-Weighted_Visual_Explanations_for_Convolutional_Neural_Networks_CVPRW_2020_paper.pdf), appeared at IEEE [CVPR 2020 Workshop on Fair, Data Efficient and Trusted Computer Vision](https://fadetrcv.github.io). Our paper has been cited by **400**!

Demo: You can run an example via [Colab](https://colab.research.google.com/drive/1m1VAhKaO7Jns5qt5igfd7lSVZudoKmID?usp=sharing)

## Update

**`2021.12.16`**: A great MATLAB implementation from [Kenta Itakura](https://github.com/KentaItakura/Explainable-AI-interpreting-the-classification-using-score-CAM).

**`2021.4.03`**: A Pytorch implementation [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (3.8K Stars).

**`2020.8.18`**: A PaddlePaddle implementation from [PaddlePaddle/InterpretDL](https://github.com/PaddlePaddle/InterpretDL).

**`2020.7.11`**: A Tensorflow implementation from [keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis).

**`2020.5.11`**: A Pytorch implementation from [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (6.2K Stars).

**`2020.3.24`**: Merged into [frgfm/torch-cam](https://github.com/frgfm/torch-cam), a wonderful library that supports multiple CAM-based methods.


## Citation
If you find this work is helpful in your research, please cite our work:
```
@inproceedings{wang2020score,
  title={Score-CAM: Score-weighted visual explanations for convolutional neural networks},
  author={Wang, Haofan and Wang, Zifan and Du, Mengnan and Yang, Fan and Zhang, Zijian and Ding, Sirui and Mardziel, Piotr and Hu, Xia},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops},
  pages={24--25},
  year={2020}
}
```

## Thanks
Utils are built on [flashtorch](https://github.com/MisaOgura/flashtorch), thanks for releasing this great work!

## Contact
If you have any questions, feel free to open an issue or directly contact me via: `haofanwang.ai@gmail.com`.
