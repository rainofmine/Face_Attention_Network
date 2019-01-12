import torch.nn as nn
import torch
import math
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors
from losses import LevelAttentionLoss, FocalLoss
from lib.nms.pth_nms import pth_nms


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations. Accept dets as tensor"""
    return pth_nms(dets, thresh)


class PyramidFeatures(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.p5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.p4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.p3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.p6 = nn.Conv2d(c5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.p7_1 = nn.ReLU()
        self.p7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs

        p5_x = self.p5_1(c5)
        p5_upsampled_x = self.p5_upsampled(p5_x)
        p5_x = self.p5_2(p5_x)

        p4_x = self.p4_1(c4)
        p4_x = p5_upsampled_x + p4_x
        p4_upsampled_x = self.p4_upsampled(p4_x)
        p4_x = self.p4_2(p4_x)

        p3_x = self.p3_1(c3)
        p3_x = p3_x + p4_upsampled_x
        p3_x = self.p3_2(p3_x)

        p6_x = self.p6(c5)

        p7_x = self.p7_1(p6_x)
        p7_x = self.p7_2(p7_x)

        return [p3_x, p4_x, p5_x, p6_x, p7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class LevelAttentionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(LevelAttentionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(feature_size, 1, kernel_size=3, padding=1)

        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.conv5(out)
        out_attention = self.output_act(out)

        return out_attention


class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers, is_cuda=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise Exception("Invalid block type")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.levelattentionModel = LevelAttentionModel(256)

        self.anchors = Anchors(is_cuda=is_cuda)

        self.regressBoxes = BBoxTransform(is_cuda=is_cuda)

        self.clipBoxes = ClipBoxes()

        self.levelattentionLoss = LevelAttentionLoss(is_cuda=is_cuda)

        self.focalLoss = FocalLoss(is_cuda=is_cuda)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # init.xavier_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.levelattentionModel.conv5.weight.data.fill_(0)
        self.levelattentionModel.conv5.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            annotations = None

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        attention = [self.levelattentionModel(feature) for feature in features]

        # i = 1
        # for level in attention:
        #     i += 1
        #     level = level.squeeze(0)
        #     level = np.array(255 * unnormalize(level)).copy()
        #     level = np.transpose(level, (1, 2, 0))
        #     plt.imsave(os.path.join('./output', str(i) + '.jpg'), level[:,:,0])

        features = [features[i] * torch.exp(attention[i]) for i in range(len(features))]

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            clc_loss, reg_loss = self.focalLoss(classification, regression, anchors, annotations)
            mask_loss = self.levelattentionLoss(img_batch.shape, attention, annotations)
            return clc_loss, reg_loss, mask_loss
        else:
            # transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes()

            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
                return [None, None, None]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.3)
            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


def resnet18(num_classes):
    return ResNet(num_classes, BasicBlock, [2, 2, 2, 2])


def resnet34(num_classes):
    return ResNet(num_classes, BasicBlock, [3, 4, 6, 3])


def resnet50(num_classes):
    return ResNet(num_classes, Bottleneck, [3, 4, 6, 3])


def resnet101(num_classes):
    return ResNet(num_classes, Bottleneck, [3, 4, 23, 3])


def resnet152(num_classes):
    return ResNet(num_classes, Bottleneck, [3, 8, 36, 3])
