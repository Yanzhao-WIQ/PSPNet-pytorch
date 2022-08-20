'''
训练文件
'''
import argparse
import os
import torch
from torch import optim
from tqdm import tqdm

from dataset import Getdata, mean_IU
from Model import Pspnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_dir', type=str, default='data/images', help='Path to images folder')
    # parser.add_argument('--mask_dir', type=str, default='data/masks', help='Path to masks folder')
    parser.add_argument('--models_dir', type=str, default='model_saved', help='saved model')
    parser.add_argument('--batch_size', type=int, default=31)
    parser.add_argument('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs to run')
    parser.add_argument('--start_lr', type=float, default=1e-3, help='start studying rate')
    parser.add_argument('--seed', type=int, default=123)
    opt = parser.parse_args()
    os.makedirs(opt.models_dir, exist_ok=True)  # 创建model_saved 文件夹
    torch.manual_seed(opt.seed)  # 生成随机种子

    model = Pspnet(num_classes=1, aux_loss=True)  # 二分类任务，输出维度num_classes=1
    model = model.to(device)

    train_loader = Getdata(train=True, batch_size=opt.batch_size)  # get train_data
    optimizer = optim.Adam(model.parameters(), lr=opt.start_lr)  # Adam
    loss_f = torch.nn.BCEWithLogitsLoss()  # 损失函数选用二进制交叉熵损失函数(含sigmoid

    for epoch in range(0, opt.epochs):
        with tqdm(total=(len(train_loader.dataset) - len(train_loader.dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.epochs))

            for data in train_loader:
                x, y = data
                x = x.to(device)
                gt = torch.sigmoid(y.unsqueeze(1).float()).to(device)
                gt = torch.where(gt > 0.5, torch.ones_like(gt), torch.zeros_like(gt))

                model.train()
                optimizer.zero_grad()
                result = model(x)
                # loss
                seg_loss = loss_f(result['output'], gt)
                cls_loss = loss_f(result['aux_output'], gt)
                loss = seg_loss + opt.alpha * cls_loss
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(loss))
                _tqdm.update(len(x))
                # MIou
                train_IU = [mean_IU(result['aux_output'][i].cpu().data.max(0)[1].detach(), gt[i, 0, :, :].cpu().data.detach(),
                            num_classes=2, ignore_index=[2 - 1]) for i in range(len(result['aux_output']))]
                train_IU = sum(train_IU) / len(train_IU)

        torch.save(model.state_dict(), os.path.join(opt.models_dir, '{}__{}.pth'.format("PSPNet", str(epoch + 1))))
        print('train MIoU:', train_IU)

