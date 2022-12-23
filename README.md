# PyTorch-Hyperbolic-YOLOv3
A minimal PyTorch implementation of YOLOv3 with hyperbolic embeddings, with support for training, inference and evaluation.Please see our slides for details about motivation and results: [Hyperbolic Object Detection](https://docs.google.com/presentation/d/129tEFTov2br3vmXDUdejqX65RqBlukDQJMmQ1tDJJSo/edit#slide=id.g1b03aa4f96f_0_0). We have two branches: branch master (original YOLOv3 that is the baseline) and branch hyperbolic_version (YOLOv3 with hyperbolic mapping and loss function).

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pytorchyolo.svg)](https://pypi.python.org/pypi/pytorchyolo/) [![PyPI license](https://img.shields.io/pypi/l/pytorchyolo.svg)](LICENSE)



## Installation

```bash
git clone https://github.com/Ys-Jia/PyTorch-Hyperbolic-YOLOv3.git
cd PyTorch-Hyperbolic-YOLOv3/
```

#### Download pretrained weights

```bash
./weights/download_weights.sh
```

#### Download COCO

```bash
./data/get_coco_dataset.sh
```

#### Download CrowdHuman ####
```bash
./data/get_crowdhuman_dataset.sh
```

#### Detect ####
detect.py will generate all detections and draw boxes for a whole folder
```
python detect.py --images "image path"
```

<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/giraffe.png" width="480"\></p>
<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/dog.png" width="480"\></p>
<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/traffic.png" width="480"\></p>
<p align="center"><img src="https://github.com/eriklindernoren/PyTorch-YOLOv3/raw/master/assets/messi.png" width="480"\></p>

## Train
Training process if fully implemented in train.py.
```
python train.py --model xxx --data xxx
```
Please specify the model config, dataset path you want to use

#### Example (COCO)
To train on COCO using a Darknet-53 backend pretrained on ImageNet run: 

```bash
python train.py --data config/coco.data  --pretrained_weights weights/darknet53.conv.74
```

## Train on Custom Dataset

#### Custom yolov3 model
Run the commands below to create a custom yolov3 model definition, replacing `<num-classes>` with the number of classes in your dataset.
Hyperbolic detector would not be affected by yolov3 config.

```bash
./config/create_custom_model.sh <num-classes>  # Will create custom model 'yolov3-custom.cfg'
```

#### Classes
Add class names to `data/custom/classes.names`. This file should have one row per class name.

#### Image Folder
Move the images of your dataset to `data/custom/images/`.

#### Annotation Folder
Move your annotations to `data/custom/labels/`. The dataloader expects that the annotation file corresponding to the image `data/custom/images/train.jpg` has the path `data/custom/labels/train.txt`. Each row in the annotation file should define one bounding box, using the syntax `label_idx x_center y_center width height`. The coordinates should be scaled `[0, 1]`, and the `label_idx` should be zero-indexed and correspond to the row number of the class name in `data/custom/classes.names`.

#### Define Train and Validation Sets
In `data/custom/train.txt` and `data/custom/valid.txt`, add paths to images that will be used as train and validation data respectively.

#### Train
To train on the custom dataset run:

```bash
python train.py --model config/yolov3-custom.cfg --data config/custom.data
```

Add `--pretrained_weights weights/darknet53.conv.74` to train using a backend pretrained on ImageNet.


## API

You are able to import the modules of this repo in your own project if you install the pip package `pytorchyolo`.

An example prediction call from a simple OpenCV python script would look like this:

```python
import cv2
from pytorchyolo import detect, models

# Load the YOLO model
model = models.load_model(
  "<PATH_TO_YOUR_CONFIG_FOLDER>/yolov3.cfg", 
  "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yolov3.weights")

# Load the image as a numpy array
img = cv2.imread("<PATH_TO_YOUR_IMAGE>")

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOLO model on the image 
boxes = detect.detect_image(model, img)

print(boxes)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]
```

For more advanced usage look at the method's doc strings.

## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

### Hyperbolic Image Embeddings
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
Computer vision tasks such as image classification, image retrieval and few-shot learning are currently dominated by Euclidean and spherical embeddings, so that the final decisions about class belongings or the degree of similarity are made using linear hyperplanes, Euclidean distances, or spherical geodesic distances (cosine similarity). In this work, we demonstrate that in many practical scenarios hyperbolic embeddings provide a better alternative.

[[Paper]](https://arxiv.org/abs/1904.02239)[[Authors' Implementation]](https://github.com/leymir/hyperbolic-image-embeddings)

```
@article{Hyperbolic Image Embeddings,
  author    = {Valentin Khrulkov and
               Leyla Mirvakhabova and
               Evgeniya Ustinova and
               Ivan V. Oseledets and
               Victor S. Lempitsky},
  title     = {Hyperbolic Image Embeddings},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.02239},
  eprinttype = {arXiv},
}
```


## Other

### YOEO — You Only Encode Once

[YOEO](https://github.com/bit-bots/YOEO) extends this repo with the ability to train an additional semantic segmentation decoder. The lightweight example model is mainly targeted towards embedded real-time applications.
