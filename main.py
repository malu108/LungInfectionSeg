import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
from datetime import datetime
from dataset import Mydata
from optimizer import RangerV2
from utils import *
from loss import ComboWithAreaLoss
import argparse
import os
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=61, help='epoch number')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_start_epoch', type=int, default=40, help='when to start to decay learning rate')
parser.add_argument('--decay_epoch', type=int, default=5, help='every n epochs decay learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=0, help='number of workers in dataloader. In windows, set num_workers=0')
parser.add_argument('--train_img_path', type=str, default='',help='images path for training')
parser.add_argument('--train_msk_path', type=str, default='',help='images mask path for training')
parser.add_argument('--valid_img_path', type=str, default='',help='images path for validing')
parser.add_argument('--valid_msk_path', type=str, default='',help='images mask path for validing')
parser.add_argument('--model_type', type=str, default='CAPARes_Unet', help='Coplenet/DDANet/segnet/U_Net/CARes_Unet/CAPARes_Unet')
parser.add_argument('--optimizer_type', type=str, default='Ranger', help='type of optimizer')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--pretrained_model', type=str, default='', help='pretrained base model')
parser.add_argument('--save_start_epoch', type=int, default=55, help='starting to save model epoch ')
parser.add_argument('--snapshots', type=int, default=5, help='every n epochs save a model')
parser.add_argument('--save_folder', type=str, default='', help='Location to save checkpoint models')

opt = parser.parse_args()
if opt.data_augmentation:
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(90),
                                          transforms.ToTensor()
                                          ])
    valid_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(90),
                                          transforms.ToTensor()
                                          ])
else:
    train_transform = transforms.Compose([transforms.ToTensor()])
    valid_transform = transforms.Compose([transforms.ToTensor()])

train_dataset = Mydata(opt.train_img_path, opt.train_msk_path, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers)
valid_dataset = MyDataset(opt.valid_img_path, opt.valid_msk_path, valid_transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers)

epo_num = opt.epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import CAPARes_Unet
model = CAPARes_Unet()

if opt.pretrained:
    if os.path.exists(opt.pretrained_model):
        model.load_state_dict(torch.load(opt.pretrained_model, map_location=lambda storage, loc: storage))
        print('Pre-trained model is loaded.')
    else:
        print("Model Not Found")
        exit(-1)
model = model.to(device)
criterion = ComboWithAreaLoss().to(device)

if opt.optimizer_type == 'Ranger':
    optimizer = RangerV2(model.parameters(), lr=opt.lr)
elif opt.optimizer_type == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer_type == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr)
print('optimizer:', opt.optimizer_type)

prev_time = datetime.now()

###############    train     ################
for epo in range(opt.start_iter, epo_num):
    train_loss = 0
    train_dice = 0
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    IoU = 0.
    model.train()
    for index, (img, img_msk) in enumerate(train_dataloader):
        img = img.to(device)
        img_msk = img_msk.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img_msk)
        dice = dice_coef_2d(output, img_msk)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_dice += dice.item()
        SE += get_sensitivity(output, img_msk)
        SP += get_specificity(output, img_msk)
        PC += get_precision(output, img_msk)
        IoU += get_iou(output, img_msk) 

    # compute time
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    prev_time = cur_time

    print('epoch: %f, train loss = %f, train dice = %f, SE = %f, SP = %f, PC = %f, IoU =%f, %s'
          % (epo, train_loss / len(train_dataloader), train_dice / len(train_dataloader), SE / len(train_dataloader),
             SP / len(train_dataloader),PC / len(train_dataloader) ,IoU / len(train_dataloader), time_str))
    if epo % opt.decay_epoch == 0 and epo > opt.decay_start_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.decay_rate
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epo >opt.save_start_epoch and epo %opt.snapshots==0:
        save_path = opt.save_folder + opt.model_type + '_epoch_{}.pth'.format(epo)
        torch.save(model.state_dict(), save_path)
        print("model_copy is saved !")

    ########### valid #########
    model.train(False)
    model.eval()
    valid_loss = 0
    valid_dice = 0
    SE = 0.  # Sensitivity (Recall)
    SP = 0.  # Specificity
    PC = 0.  # Precision
    IoU = 0.
    for index, (img, img_msk) in enumerate(valid_dataloader):
        img = img.to(device)
        img_msk = img_msk.to(device)

        output = model(img)
        loss = criterion(output, img_msk)
        dice = dice_coef_2d(output, img_msk)

        valid_loss += loss.item()
        valid_dice += dice.item()
        SE += get_sensitivity(output, img_msk)
        SP += get_specificity(output, img_msk)
        PC += get_precision(output, img_msk)
        IoU += get_iou(output, img_msk)

    print('epoch: %f, valid loss = %f, valid dice = %f, SE = %f, SP = %f, PC = %f, IoU =%f'
          % (epo, valid_loss / len(valid_dataloader), valid_dice / len(valid_dataloader), SE / len(valid_dataloader),
             SP / len(valid_dataloader), PC / len(valid_dataloader), IoU / len(valid_dataloader)))
