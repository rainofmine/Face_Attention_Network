import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.utils.model_zoo as model_zoo

from model_level_attention import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.utils.data import DataLoader
from csv_eval import evaluate
from dataloader import WIDERDataset, AspectRatioBasedSampler, collater, Resizer, Augmenter, Normalizer, CSVDataset

is_cuda = torch.cuda.is_available()
print('CUDA available: {}'.format(is_cuda))

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

ckpt = False


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--wider_train', help='Path to file containing WIDER training annotations (see readme)')
    parser.add_argument('--wider_val',
                        help='Path to file containing WIDER validation annotations (optional, see readme)')
    parser.add_argument('--wider_train_prefix', help='Prefix path to WIDER train images')
    parser.add_argument('--wider_val_prefix', help='Prefix path to WIDER validation images')

    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--batch_size', help='Batch size (default 2)', type=int, default=2)

    parser.add_argument('--model_name', help='name of the model to save')
    parser.add_argument('--pretrained', help='pretrained model name in weight directory')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.wider_train is None:
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Resizer(), Augmenter(), Normalizer()]))
    else:
        dataset_train = WIDERDataset(train_file=parser.wider_train, img_prefix=parser.wider_train_prefix,
                                     transform=transforms.Compose([Resizer(), Augmenter(), Normalizer()]))

    if parser.wider_val is None:
        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            print('Loading CSV validation dataset')
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Resizer(), Normalizer()]))
    else:
        print('Loading WIDER validation dataset')
        dataset_val = WIDERDataset(train_file=parser.wider_val, img_prefix=parser.wider_val_prefix,
                                   transform=transforms.Compose([Resizer(), Normalizer()]))

    print('Loading training dataset')
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=16, collate_fn=collater, batch_sampler=sampler)

    # Create the model_pose_level_attention
    if parser.depth == 18:
        retinanet = resnet18(num_classes=dataset_train.num_classes())
    elif parser.depth == 34:
        retinanet = resnet34(num_classes=dataset_train.num_classes())
    elif parser.depth == 50:
        retinanet = resnet50(num_classes=dataset_train.num_classes())
    elif parser.depth == 101:
        retinanet = resnet101(num_classes=dataset_train.num_classes())
    elif parser.depth == 152:
        retinanet = resnet152(num_classes=dataset_train.num_classes())
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if ckpt:
        retinanet = torch.load('')
        print('Loading checkpoint')
    else:
        print('Loading pretrained model')
        retinanet_dict = retinanet.state_dict()
        if parser.pretrained is None:
            pretrained_dict = model_zoo.load_url(model_urls['resnet' + str(parser.depth)])
        else:
            pretrained_dict = torch.load('./weight/' + parser.pretrained)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in retinanet_dict}
        retinanet_dict.update(pretrained_dict)
        retinanet.load_state_dict(retinanet_dict)
        print('load pretrained backbone')

    print(retinanet)
    retinanet = torch.nn.DataParallel(retinanet, device_ids=[0])
    retinanet.cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    # optimizer = optim.SGD(retinanet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    f_map = open('./mAP_txt/' + parser.model_name + '.txt', 'a')
    writer = SummaryWriter(log_dir='./summary')
    iters = 0
    for epoch_num in range(0, parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):

            iters += 1

            optimizer.zero_grad()

            classification_loss, regression_loss, mask_loss = retinanet(
                [data['img'].cuda().float(), data['annot'].cuda()])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            mask_loss = mask_loss.mean()

            loss = classification_loss + regression_loss + mask_loss

            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | '
                'mask_loss {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), float(mask_loss),
                    np.mean(loss_hist)))

            writer.add_scalar('classification_loss', float(classification_loss), iters)
            writer.add_scalar('regression_loss', float(regression_loss), iters)
            writer.add_scalar('loss', float(loss), iters)

            del classification_loss
            del regression_loss
            del loss

        if parser.wider_val is not None:
            print('Evaluating dataset')

            mAP = evaluate(dataset_val, retinanet)
            f_map.write('mAP:{}, epoch:{}'.format(mAP[0][0], epoch_num))
            f_map.write('\n')

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, './ckpt/' + parser.model_name + '_{}.pt'.format(epoch_num))

    retinanet.eval()

    writer.export_scalars_to_json("./summary/' + parser.pretrained + 'all_scalars.json")
    f_map.close()
    writer.close()


if __name__ == '__main__':
    main()
