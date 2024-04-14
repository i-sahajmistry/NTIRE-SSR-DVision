## This repo contains the DVision team's solution to NTIRE Stereo Image Super Resolution Challenge

>Note: Team DVision (NTIRE 2024) and Team LongClaw (NTIRE 2023)

The official NAFNet implementation **[GitHub](https://github.com/megvii-research/NAFNet)**. 

Original Paper **[Simple Baselines for Image Restoration (ECCV2022)](https://arxiv.org/abs/2204.04676)**

#### Sahaj K. Mistry, Aryan, Sourav Saini, Aashray Gupta

>The aim of this challenge was to obtain SR results with the highest fidelity score (i.e., PSNR) under standard Bicubic degradation. In this challenge, the model size (i.e., number of parameters) was restricted to 1 M, and the computational complexity (i.e., number of MACs) was restricted to 400 G (a stereo image pair of size 320 × 180). 


### Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [HINet](https://github.com/megvii-model/HINet) 

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

```
git clone https://github.com/megvii-research/NAFNet
cd NAFNet
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

### Quick Start 

Change the `--opt` file as required. `options/train/NTIRE/NTIRE-<23/24>.yml`

* Stereo Image Inference Demo:

    * Training:
        
        * `model_type: KnowledgeDistillationModel` To use KnowledgeDistillation.
        * `odconv: true` In `network_g` to use ODConv instead of Conv.

    ```bash
    python basicsr/train.py -opt options/train/NTIRE/NITRE-24.yml
    ```

    * Testing:
    ```bash
    python basicsr/test.py -opt options/test/NAFSSR/NAFSSR-T_x4.yml
    ```

    * Inference:

    ```bash
    python basicsr/demo_ssr.py -opt options/test/NAFSSR/NAFSSR-L_4x.yml \
    --input_path <Path to folder containing input images> \
    --output_path <Path to folder to write output images>
    ```


### Citations
If DVision's version helps your research or work, please consider citing NTIRE 2024 SSR Challenge Paper.

<details>

<summary>NTIRE-24</summary>

```
To be updated!
```

</details>

<details>

<summary>NTIRE-23</summary>

```
@INPROCEEDINGS{10208910,
  author={Wang, Longguang and Guo, Yulan and Wang, Yingqian and Li, Juncheng and Gu, Shuhang and Timofte, Radu and Cheng, Ming and Ma, Haoyu and Ma, Qiufang and Sun, Xiaopeng and Zhao, Shijie and Sheng, Xuhan and Ding, Yukan and Sun, Ming and Wen, Xing and Zhang, Dafeng and Li, Jia and Wang, Fan and Xie, Zheng and He, Zongyao and Qiu, Zidian and Pan, Zilin and Zhan, Zhihao and Xian, Xingyuan and Jin, Zhi and Zhou, Yuanbo and Deng, Wei and Nie, Ruofeng and Zhang, Jiajun and Gao, Qinquan and Tong, Tong and Zhang, Kexin and Zhang, Junpei and Peng, Rui and Ma, Yanbiao and Jiao, Licheng and Bai, Haoran and Kong, Lingshun and Pan, Jinshan and Dong, Jiangxin and Tang, Jinhui and Cao, Pu and Huang, Tianrui and Yang, Lu and Song, Qing and Chen, Bingxin and He, Chunhua and Chen, Meiyun and Guo, Zijie and Luo, Shaojuan and Cao, Chengzhi and Wang, Kunyu and Zhang, Fanrui and Zhang, Qiang and Mehta, Nancy and Murala, Subrahmanyam and Dudhane, Akshay and Wang, Yujin and Li, Lingen and Gendy, Garas and Sabor, Nabil and Hou, Jingchao and He, Guanghui and Chen, Junyang and Li, Hao and Shi, Yukai and Yang, Zhijing and Zou, Wenbin and Zhang, Yunchen and Jiang, Mingchao and Yu, Zhongxin and Tan, Ming and Gao, Hongxia and Luo, Ziwei and Gustafsson, Fredrik K. and Zhao, Zheng and Sjölund, Jens and Schön, Thomas B. and Chen, Jingxiang and Yang, Bo and Zhang, XiSheryl and Li, Chenghua and Yuan, Weijun and Li, Zhan and Deng, Ruting and Zeng, Jintao and Mahajan, Pulkit and Mistry, Sahaj and Chatterjee, Shreyas and Jakhetiya, Vinit and Subudhi, Badri and Jaiswal, Sunil and Zhang, Zhao and Zheng, Huan and Zhao, Suiyi and Gao, Yangcheng and Wei, Yanyan and Wang, Bo and Li, Gen and Li, Aijin and Sun, Lei and Chen, Ke and Tang, Congling and Li, Yunzhe and Chen, Jun and Chiang, Yuan-Chun and Chen, Yi-Chung and Huang, Zhi-Kai and Yang, Hao-Hsiang and Chen, I-Hsiang and Kuo, Sy-Yen and Wang, Yiheng and Zhu, Gang and Yang, Xingyi and Liu, Songhua and Jing, Yongcheng and Hu, Xingyu and Song, Jianwen and Sun, Changming and Sowmya, Arcot and Park, Seung Ho and Lei, Xiaoyan and Wang, Jingchao and Zhai, Chenbo and Zhang, Yufei and Cao, Weifeng and Zhang, Wenlong},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)}, 
  title={NTIRE 2023 Challenge on Stereo Image Super-Resolution: Methods and Results}, 
  year={2023},
  volume={},
  number={},
  pages={1346-1372},
  keywords={Degradation;Computer vision;Conferences;Superresolution;Transformers;Distortion;Data augmentation},
  doi={10.1109/CVPRW59228.2023.00141}}
```

</details>

<details>

<summary>Also please consider citing NAFSSR.</summary>

```
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239-1248}
}
```

</details>


### Contact

If you have any questions, please contact sahajmistry005@gmail.com

---

