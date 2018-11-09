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
pip install cffi
pip install cython
pip install pandas
pip install tensorboardX
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
You should prepare three CSV or TXT files including train annotations file, valid annotations file and label encoding file. 

### Annotations format
Two examples are as follows:

```
$image_path/img_1.jpg x1 y1 x2 y2 label
$image_path/img_2.jpg . . . . .
```

Images with more than one bounding box should use one row per box. When an image does not contain any bounding box, set them '.'. 

### Label encoding file
A TXT file (classes.txt) is needed to map label to ID. Each line means one label name and its ID. One example is as follows:

```
face 0
```

## Pretrained Model

We use resnet18, 34, 50, 101, 152 as the backbone. You should download them and put them to `/weight`.

- resnet18: [https://download.pytorch.org/models/resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth)
- resnet34: [https://download.pytorch.org/models/resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth)
- resnet50: [https://download.pytorch.org/models/resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)
- resnet101: [https://download.pytorch.org/models/resnet101-5d3b4d8f.pth](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)
- resnet152: [https://download.pytorch.org/models/resnet152-b121ed2d.pth](https://download.pytorch.org/models/resnet152-b121ed2d.pth)

## Training

```
python train.py --csv_train <$path/train.txt> --csv_val <$path/val.txt> --csv_classes <$path/classes.txt> --depth <50> --pretrained resnet50-19c8e357.pth --model_name <model name to save>
```

## Visualization Result
Detection result

![img2](https://github.com/rainofmine/face_attention_network/blob/master/img/2.png)

Attention map at different level (P3~P7)

![img3](https://github.com/rainofmine/face_attention_network/blob/master/img/3.png)

## Reference

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Face Attention Network: An Effective Face Detector for the Occluded Faces](https://arxiv.org/abs/1711.07246)