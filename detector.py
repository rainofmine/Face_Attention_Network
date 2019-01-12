import cv2
import skimage
import argparse
import numpy as np
import torch
import json

from torchvision import transforms

from dataloader import Resizer, Normalizer


def _load_image(img_name):
    img = cv2.imread(img_name)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    return img.astype(np.float32) / 255.0


def fan_detect(model, img_name, threshold=0.9, max_detections=100, is_cuda=True):
    input_data = {'img': _load_image(img_name), 'annot': np.zeros((0, 5)), 'scale': 1}
    transform = transforms.Compose([Resizer(), Normalizer()])
    transformed = transform(input_data)

    model.eval()
    with torch.no_grad():
        img_data = transformed['img'].permute(2, 0, 1).float().unsqueeze(dim=0)
        if is_cuda:
            img_data = img_data.cuda()
        scores, labels, boxes = model(img_data)
        scores = scores.numpy()
        scale = transformed['scale']
        boxes = boxes.numpy() / scale

        indices = np.where(scores > threshold)[0]
        scores = scores[indices]
        scores_sort = np.argsort(-scores)[:max_detections]
        image_boxes = boxes[indices[scores_sort], :]

    return image_boxes


def img_rectangles(img_path, output_path, boxes=None):
    img = cv2.imread(img_path)

    if boxes is not None:
        for arr in boxes:
            cv2.rectangle(img, (arr[0], arr[1]), (arr[2], arr[3]), (0, 0, 255), 1)

    cv2.imwrite(output_path, img)


def load_model(model_path, is_cuda=True):
    # load possible cuda model as cpu
    model = torch.load(model_path, map_location=lambda storage, location: storage)
    if is_cuda:
        model = model.cuda()
    return model


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--model', help='Path to model')
    parser.add_argument('--image', help='Path to image')
    parser.add_argument('--rect', help='Output image with rectangles')
    parser.add_argument('--threshold', help='Probability threshold (default 0.9)', type=float, default=0.9)
    parser.add_argument('--force-cpu', help='Force CPU for detection (default false)', dest='force_cpu',
                        default=False, action='store_true')

    parser = parser.parse_args(args)

    is_cuda = torch.cuda.is_available() and not parser.force_cpu

    model = load_model(parser.model, is_cuda=is_cuda)
    boxes = fan_detect(model, parser.image, threshold=parser.threshold, is_cuda=is_cuda)
    print(json.dumps(boxes))
    if parser.rect:
        img_rectangles(parser.image, parser.rect, boxes)


if __name__ == '__main__':
    main()
