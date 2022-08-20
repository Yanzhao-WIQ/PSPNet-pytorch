'''
测试文件
'''
import argparse
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
import numpy as np

from dataset import Getdata, mean_IU
from Model import Pspnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_dir', type=str, default='data/images', help='Path to images folder')
    # parser.add_argument('--mask_dir', type=str, default='data/masks', help='Path to masks folder')
    # parser.add_argument('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
    # parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs to run')
    # parser.add_argument('--start_lr', type=float, default=1e-3, help='start studying rate')
    parser.add_argument('--batch_size', type=int, default=49)
    parser.add_argument('--models_dir', type=str, default='model_saved/PSPNet__5.pth', help='saved model')
    parser.add_argument('--seed', type=int, default=123)
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)  # 生成随机种子

    model = Pspnet(num_classes=1, aux_loss=True)  # 二分类任务，输出维度num_classes=1

    # load model
    state_dict = model.state_dict()
    for n, p in torch.load(opt.models_dir, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    val_loader = Getdata(train=False, batch_size=opt.batch_size)  # get val_data

    for data in val_loader:
        x, y = data
        x = x.to(device)
        gt = torch.sigmoid(y.unsqueeze(1).float()).to(device)
        gt = torch.where(gt > 0.5, torch.ones_like(gt), torch.zeros_like(gt))

        result = model(x)

        output = torch.sigmoid(result['output'])  # net的最后一层没有使用sigmoid
        aux_output = torch.sigmoid(result['aux_output'])
        aux_mean = [aux_output[i].squeeze(1).detach().numpy().mean() for i in range(len(aux_output))]
        sum_aux_mean = sum(aux_mean) / len(aux_mean)
        predict = torch.where(aux_output > sum_aux_mean, torch.ones_like(aux_output), torch.zeros_like(aux_output))
        # MIou
        val_IU = [mean_IU(result['aux_output'][i].cpu().data.max(0)[1].detach(), gt[i, 0, :, :].cpu().data.detach(),
                          num_classes=2, ignore_index=[2 - 1]) for i in range(len(result['aux_output']))]
        val_IU = sum(val_IU) / len(val_IU)
        print('val MIoU:', val_IU)

        # save result image
        '''
        for i in range(len(val_loader.dataset)):
            save_image(gt[i], "data/val_result/{}G_{}.png".format(i, 'gt'))
            save_image(result['output'][i], "data/val_result/{}_{}A.png".format(i, 'output'))
            save_image(predict[i], "data/val_result/{}_{}PRE.png".format(i, 'aux_output'))
            save_image(aux_output[i], "data/val_result/{}_{}Sig.png".format(i, 'aux_output'))
            save_image(result['aux_output'][i], "data/val_result/{}_{}A.png".format(i, 'aux_output'))
            save_image(output[i], "data/val_result/{}_{}Sig.png".format(i, 'output'))

            plt.axis('off')
            plt.imshow(np.squeeze(result['output'][i].detach().numpy()))  # transfer tensor to array
            plt.savefig("data/val_result/{}_{}.png".format(i, 'output'), bbox_inches='tight', dpi=450)

            plt.imshow(np.squeeze(result['aux_output'][i].detach().numpy()))  # transfer tensor to array
            plt.savefig("data/val_result/{}_{}.png".format(i, 'aux_output'), bbox_inches='tight', dpi=450)
        '''
