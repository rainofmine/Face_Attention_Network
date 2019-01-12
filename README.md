# Face Attention Network

Pytorch implementation of face attention network as described in [Face Attention Network: An Effective Face Detector for the Occluded Faces](https://arxiv.org/abs/1711.07246). The baseline is RetinaNet followed by this [repo](https://github.com/yhenon/pytorch-retinanet).

![img1](https://github.com/rainofmine/face_attention_network/blob/master/img/1.png)

## Requirements

- Python3
- Pytorch0.4
- torchvision
- tensorboardX

## Installation

Install packages.

```
sudo apt-get install tk-dev python-tk
pip install -r requirements.txt
```

Build NMS.

```
cd Face_Attention_Network/lib
sh build.sh
```

Create folders.

```
cd Face_Attention_Network/
mkdir ckpt mAP_txt summary weight
```

## Datasets

### CSV dataset
You should prepare three CSV or TXT files including train annotations file, valid annotations file and label encoding file.

#### Annotations format
Two examples are as follows:

```
$image_path/img_1.jpg x1 y1 x2 y2 label
$image_path/img_2.jpg . . . . .
```

Images with more than one bounding box should use one row per box. When an image does not contain any bounding box, set them '.'. 

#### Label encoding file
A TXT file (classes.txt) is needed to map label to ID. Each line means one label name and its ID. One example is as follows:

```
face 0
```

### WIDER dataset

Download WIDER dataset from [http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/).
You need training and validation images and face annotations. 

## Pretrained Model

We use resnet18, 34, 50, 101, 152 as the backbone. You should also use your own pretrained model in `/weight` directory.

## Training

### CSV dataset

```
python train.py --csv_train <$path/train.txt> --csv_val <$path/val.txt> --csv_classes <$path/classes.txt> --depth <50> --model_name <model name to save>
```

### WIDER dataset

```
python train.py --wider_train <$path/wider_face_train_bbx_gt.txt> --wider_val <$path/wider_face_val_bbx_gt.txt> --wider_train_prefix <$path/WIDER_train/images> --wider_val_prefix <$path/WIDER_val/images> --depth <50> --model_name <model name to save>
```

## Visualization Result
Detection result

![img2](https://github.com/rainofmine/face_attention_network/blob/master/img/2.png)

Attention map at different level (P3~P7)

![img3](https://github.com/rainofmine/face_attention_network/blob/master/img/3.png)

## Reference

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Face Attention Network: An Effective Face Detector for the Occluded Faces](https://arxiv.org/abs/1711.07246)