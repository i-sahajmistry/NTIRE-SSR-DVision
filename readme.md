## This repo contains the DVision team's solution to NTIRE 2024 Stereo Image Super Resolution Challenge

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

>Change the `--opt` file as required.

* Stereo Image Inference Demo:

    * Training:
    ```bash
    python basicsr/train.py -opt options/train/NAFSSR-KD/NAFSSR-T_SwinFIRSSR_x4.yml
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

```
To be Updated!
```

Also please consider citing NAFSSR.
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

### Contact

If you have any questions, please contact sahajmistry005@gmail.com

---

