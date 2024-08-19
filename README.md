# HSEmotionONNX Python Library for Facial Emotion Recognition
[![Downloads](https://static.pepy.tech/personalized-badge/hsemotion_onnx?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20installs)](https://pepy.tech/project/hsemotion_onnx)
[![pypi package](https://img.shields.io/badge/version-v0.3.1-blue)](https://pypi.org/project/hsemotion_onnx)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classifying-emotions-and-engagement-in-online/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=classifying-emotions-and-engagement-in-online)

## Prerequisites

- python>=3.9


## Installing dependencies

```
python setup.py install
```

It is also possible to install it via pip:
```
pip install -e .
```

## Usage

### Test RTSP stream

檢查RTSP串流是否正常

```
python hsemotion_onnx/check_rtsp_with_opencv.py --url rtsp://b03773d78e34.entrypoint.cloud.wowza.com:1935/app-4065XT4Z/80c76e59_stream1

```

### Emotion Recognition

- 使用webcam

```
python hsemotion_onnx/facial_emotions_demo.py
```

- 使用RTSP串流 (單個)

**注意**：可以使用`--skip-frame`參數來跳過幾個frame

```
python hsemotion_onnx/facial_emotions_demo.py --rtsp-url rtsp://simplenoodle:######@192.168.1.xx:yyy/Streaming/Channels/101 --skip-frame 13
```

### FastAPI

開啟fastapi

```
cd api
uvicorn main:app --reload
```

swagger ui

```
http://127.0.0.1:8000/docs
```

## License

This project uses the following open-source libraries:

- [onnxruntime](https://github.com/microsoft/onnxruntime) - Licensed under the MIT License.
- [fastapi](https://github.com/fastapi/fastapi) - Licensed under the MIT License.
- [screeninfo](https://github.com/rr-/screeninfo?tab=License-1-ov-file) - Licensed under the MIT License.
- [hsemotion-onnx](https://github.com/av-savchenko/hsemotion-onnx?tab=readme-ov-file) - Licensed under the Apache-2.0 License.
- [onnx](https://github.com/onnx/onnx) - Licensed under the Apache-2.0 License.
- [opencv-python](https://github.com/opencv/opencv-python) - Licensed under the Apache-2.0 License.
- [pymongo](https://github.com/mongodb/mongo-python-driver) - Licensed under the Apache-2.0 License.
- [numpy](https://github.com/numpy/numpy) - Licensed under the BSD-3-Clause License.
- [uvicorn](https://github.com/encode/uvicorn) - Licensed under the BSD-3-Clause License.

## Citation

numpy

```BibTex
@ARTICLE{2020NumPy-Array,
  author  = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, Stéfan J and Gommers, Ralf and Virtanen, Pauli and Cournapeau, David and Wieser, Eric and Taylor, Julian and Berg, Sebastian and Smith, Nathaniel J. and Kern, Robert and Picus, Matti and Hoyer, Stephan and van Kerkwijk, Marten H. and Brett, Matthew and Haldane, Allan and Fernández del Río, Jaime and Wiebe, Mark and Peterson, Pearu and Gérard-Marchant, Pierre and Sheppard, Kevin and Reddy, Tyler and Weckesser, Warren and Abbasi, Hameer and Gohlke, Christoph and Oliphant, Travis E.},
  title   = {Array programming with {NumPy}},
  journal = {Nature},
  year    = {2020},
  volume  = {585},
  pages   = {357–362},
  doi     = {10.1038/s41586-020-2649-2}
}
```

opencv-python

```BibTex
@article{opencv_library,
    author = {Bradski, G.},
    citeulike-article-id = {2236121},
    journal = {Dr. Dobb's Journal of Software Tools},
    keywords = {bibtex-import},
    posted-at = {2008-01-15 19:21:54},
    priority = {4},
    title = {{The OpenCV Library}},
    year = {2000}
}
```

onnxruntime

```BibTex
@software{ONNX_Runtime_developers_ONNX_Runtime_2018,
author = {{ONNX Runtime developers}},
license = {MIT},
month = nov,
title = {{ONNX Runtime}},
url = {https://github.com/microsoft/onnxruntime},
year = {2018}
}
```

fastapi

```BibTex
@software{Ramirez_FastAPI,
author = {Ramírez, Sebastián},
license = {MIT},
title = {{FastAPI}},
url = {https://github.com/fastapi/fastapi}
}
```

centerface

```BibTex
@inproceedings{CenterFace,
  title={CenterFace: Joint Face Detection and Alignment Using Face as Point},
  author={Xu, Yuanyuan and Yan, Wan and Sun, Haixin and Yang, Genke and Luo, Jiliang},
  booktitle={arXiv:1911.03599},
  year={2019}
}
```

hsemotion

```BibTex
@inproceedings{savchenko2023facial,
  title = 	 {Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction},
  author =       {Savchenko, Andrey},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  pages = 	 {30119--30129},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  url={https://proceedings.mlr.press/v202/savchenko23a.html}
}
```

```BibTex
@inproceedings{savchenko2021facial,
  title={Facial expression and attributes recognition based on multi-task learning of lightweight neural networks},
  author={Savchenko, Andrey V.},
  booktitle={Proceedings of the 19th International Symposium on Intelligent Systems and Informatics (SISY)},
  pages={119--124},
  year={2021},
  organization={IEEE},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@inproceedings{Savchenko_2022_CVPRW,
  author    = {Savchenko, Andrey V.},
  title     = {Video-Based Frame-Level Facial Analysis of Affective Behavior on Mobile Devices Using EfficientNets},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2022},
  pages     = {2359-2366},
  url={https://arxiv.org/abs/2103.17107}
}
```

```BibTex
@inproceedings{Savchenko_2022_ECCVW,
  author    = {Savchenko, Andrey V.},
  title     = {{MT-EmotiEffNet} for Multi-task Human Affective Behavior Analysis and Learning from Synthetic Data},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV 2022) Workshops},
  pages={45--59},
  year={2023},
  organization={Springer},
  url={https://arxiv.org/abs/2207.09508}
}
```


```BibTex
@article{savchenko2022classifying,
  title={Classifying emotions and engagement in online learning based on a single facial expression recognition neural network},
  author={Savchenko, Andrey V and Savchenko, Lyudmila V and Makarov, Ilya},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE},
  url={https://ieeexplore.ieee.org/document/9815154}
}
```

### hsemotion_onnx part

```
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    model_name='enet_b0_8_best_afew'
    fer=HSEmotionRecognizer(model_name=model_name)
    emotion,scores=fer.predict_emotions(face_img,logits=False)
```

The following values of `model_name` parameter are supported:
- enet_b0_8_best_vgaf
- enet_b0_8_best_afew
- enet_b0_8_va_mtl
- enet_b2_8
- enet_b2_7

The method `predict_emotions` returns both the string value of predicted emotions (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, or Surprise) and scores at the output of the last layer. 
If the `logits` parameter is set to `True` (by default), the logits are returned, otherwise, the posterior probabilities are estimated from the logits using softmax.


The versions of this method for a batch of images are also available
```
    emotions,scores=fer.predict_multi_emotions(face_img_list,logits=False)
```

Complete usage examples are available in the [demo folder](demo). It is necessary to install [mediapipe](https://google.github.io/mediapipe/) to run the demo script.

The details about training of the models are available in the [main repository](https://github.com/HSE-asavchenko/face-emotion-recognition)