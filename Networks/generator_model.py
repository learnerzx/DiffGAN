""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import pickle
from scipy.io import loadmat, savemat
from vvt.vvt import vvt_tiny
from .unet_parts import *
import cv2
import scipy
# import vision_transformer as SwinUnet
# from Masks.mask_generator import Gaussian2DMaskFunc, RadialMaskFunc, SpiralMaskFunc,VariableDensityPoissonMaskFunc,FastMRIRandomMaskFunc
from .vision_transformer import *
class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):  #频域生成器初始参数 (self,6,2,True) 图像域初始参数 (self,1,1,True)
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """

        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 1 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

        # self.vvt_down=vvt_tiny(in_chans=32)

        # self.vvt_down2=vvt_tiny(in_chans=64, num_classes=128)
        # self.vvt_down3=vvt_tiny(in_chans=128, num_classes=256)
        # self.vvt_down4=vvt_tiny(in_chans=256, num_classes=512 // factor)

    def forward(self, x):


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # x1 = self.inc(x)
        # _,outs = self.vvt_down(x1)
        # x3 = self.vvt_down2(x2)
        # x4 = self.vvt_down3(x3)
        # x5 = self.vvt_down4(x4)

        # x = self.up1(outs[3], outs[2])
        # x = self.up2(x, outs[1])
        # x = self.up3(x, outs[0])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class LVT_UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=True):  #频域生成器初始参数 (self,6,2,True) 图像域初始参数 (self,1,1,True)
        """U-Net  #https://github.com/milesial/Pytorch-UNet
        """

        super(LVT_UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels_in, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 1 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_channels_out)

        self.vvt_down=vvt_tiny(in_chans=32)


    def forward(self, x):

        x1 = self.inc(x)
        _,outs = self.vvt_down(x1)
        x = self.up1(outs[3], outs[2])
        x = self.up2(x, outs[1])
        x = self.up3(x, outs[0])
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

class WNet(nn.Module):

    def __init__(self, args):
        super(WNet, self).__init__()

        self.bilinear = args.bilinear
        self.args = args
        self.masked_kspace = args.masked_kspace

        # if args.mask_type == 'Gaussian2D':
        #     self.mask_func = Gaussian2DMaskFunc(accelerations=args.accelerations, center_fractions=args.center_fractions)
        # elif args.mask_type == 'Radial':
        #     self.mask_func = RadialMaskFunc(accelerations=args.accelerations, center_fractions=args.center_fractions)
        # elif args.mask_type == 'Spiral':
        #     self.mask_func = SpiralMaskFunc(accelerations=args.accelerations, center_fractions=args.center_fractions)
        # elif args.mask_type == 'Poisson':
        #     self.mask_func = VariableDensityPoissonMaskFunc(accelerations=args.accelerations, center_fractions=args.center_fractions)
        # elif args.mask_type == 'Random':
        #     self.mask_func = FastMRIRandomMaskFunc(accelerations=args.accelerations, center_fractions=args.center_fractions)
        # else:
        #     raise ValueError('Unknown mask type {}'.format(args.mask_type))

        # mask_path = args.mask_path

        # if args.mask_type=='radial' and args.sampling_percentage==30:
        #     with open(mask_path, 'rb') as pickle_file:
        #         masks = pickle.load(pickle_file)
        #         # self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        #         self.mask = torch.tensor(masks, device=self.args.device).float()
        # elif args.mask_type=='radial' and args.sampling_percentage==50:
        #     mask_shift = cv2.imread(r'E:\code\code_backup\Masks\radial\radial_50.tif', 0) / 255
        #     # mask = scipy.ifft(mask_shift)
        #     # mask_shift= self.fftshift(mask_shift)
        #     self.mask = torch.tensor(mask_shift == 1, device=self.args.device)
        # elif args.mask_type=='random':
        #     with open(mask_path, 'rb') as pickle_file:
        #         masks = pickle.load(pickle_file)
        #         self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        # else:
        #     masks = loadmat(r'E:\code\code_backup_vvt+diffusion\Masks_new\Radial_[4].mat')
        #     try:
        #         self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)
        #     except:
        #         try:
        #             self.mask = torch.tensor(masks['maskRS2'] == 1, device=self.args.device)
        #         except:
        #             self.mask = torch.tensor(masks['population_matrix'] == 1, device=self.args.device)


        mask_path='./Masks_new/{}/{}.mat'.format(args.mask_type, args.accelerations)
        masks = loadmat(mask_path)
        self.mask = torch.tensor(masks['mask1'] == 1, device=self.args.device)

        self.maskNot = self.mask == 0

        if self.args.ST:
            # self.kspace_Unet = SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes + 1,in_chans=6).cuda()
            self.kspace_Unet = LVT_UNet(n_channels_in=args.num_input_slices*2, n_channels_out=2, bilinear=self.bilinear)
            # self.kspace_Unet = UNet(n_channels_in=args.num_input_slices*2, n_channels_out=2, bilinear=self.bilinear)
            # self.img_UNet = SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes, in_chans=1).cuda()
            # self.img_UNet = UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)
            self.img_UNet = LVT_UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)

        else:
            #调用k空间U-Net和图像域U-Net
            self.kspace_Unet = UNet(n_channels_in=args.num_input_slices*2, n_channels_out=2, bilinear=self.bilinear)
            # self.img_UNet = UNet(n_channels_in=1, n_channels_out=1, bilinear=self.bilinear)

            # self.kspace_Unet = SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes+1,in_chans=6).cuda()
            self.img_UNet=SwinUnet(args, img_size=args.img_size, num_classes=args.num_classes,in_chans=1).cuda()

    def fftshift(self, img):

        S = int(img.shape[3]/2)
        img2 = torch.zeros_like(img)
        img2[:, :, :S, :S] = img[:, :, S:, S:]
        img2[:, :, S:, S:] = img[:, :, :S, :S]
        img2[:, :, :S, S:] = img[:, :, S:, :S]
        img2[:, :, S:, :S] = img[:, :, :S, S:]
        return img2

    def inverseFT(self, Kspace):
        Kspace = Kspace.permute(0, 2, 3, 1)
        img_cmplx = torch.ifft(Kspace, 2)
        img = torch.sqrt(img_cmplx[:, :, :, 0]**2 + img_cmplx[:, :, :, 1]**2)
        img = img[:, None, :, :]
        return img

    def forward(self, Kspace):

        rec_all_Kspace = self.kspace_Unet(Kspace)

        if self.masked_kspace:
##################################################################################################

            # self.mask=self.mask_func([256, 256, 1],seed=self.args.seed).permute(3, 0, 1, 2).cuda()
            # # self.mask=torch.cat([self.mask,self.mask],dim=1).cuda()
            # self.maskNot = self.mask == 0
            # self.maskNot = self.maskNot.cuda()
##################################################################################################
            rec_Kspace = self.mask*Kspace[:, int(Kspace.shape[1]/2)-1:int(Kspace.shape[1]/2)+1, :, :] +\
                         self.maskNot*rec_all_Kspace
            rec_mid_img = self.inverseFT(rec_Kspace)

        else:
            rec_Kspace = rec_all_Kspace
            rec_mid_img = self.inverseFT(rec_Kspace)            ##  频域生成图像 rec_mid_img


        Img = self.inverseFT(Kspace[:,4:6,:,:])
        refine_Img = self.img_UNet(Img)                             ##  图像域生成图像
        # rec_img =rec_mid_img
        # rec_img = torch.tanh(rec_mid_img)
        # rec_img =refine_Img
        # rec_img = torch.tanh(refine_Img )
        rec_img = torch.tanh(refine_Img +rec_mid_img )


        # rec_img= self.data_normal_2d(torch.squeeze(rec_img))

        rec_img = torch.clamp(rec_img, 0, 1)
        # if self.train():
        return rec_img, rec_Kspace, rec_mid_img

    def data_normal_2d(self,orign_data, dim="col"):
        """
        针对于2维tensor归一化
        可指定维度进行归一化，默认为行归一化
        参数1为原始tensor，参数2为默认指定行，输入其他任意则为列
        """
        out_put_data=[]
        for i in range(orign_data.shape[0]):
            if dim == "col":
                dim = 1
                d_min = torch.min(orign_data[i], dim=dim)[0]
                for idx, j in enumerate(d_min):
                    if j < 0:
                        orign_data[i,idx, :] += torch.abs(d_min[idx])
                        d_min = torch.min(orign_data[i], dim=dim)[0]
            else:
                dim = 0
                d_min = torch.min(orign_data[i], dim=dim)[0]
                for idx, j in enumerate(d_min):
                    if j < 0:
                        orign_data[i,idx, :] += torch.abs(d_min[idx])
                        d_min = torch.min(orign_data[i], dim=dim)[0]
            d_max = torch.max(orign_data[i], dim=dim)[0]
            dst = d_max - d_min
            if d_min.shape[0] == orign_data[i].shape[0]:
                d_min = d_min.unsqueeze(1)
                dst = dst.unsqueeze(1)
            else:
                d_min = d_min.unsqueeze(0)
                dst = dst.unsqueeze(0)
            norm_data = torch.sub(orign_data[i], d_min).true_divide(dst)
            out_put_data.append(norm_data)
        out_put_data = torch.tensor([item.cpu().detach().numpy() for item in out_put_data]).cuda()

        return torch.unsqueeze(out_put_data,1)

    def linear_scale(self,img):
        img = img - np.min(img)
        img = img / np.max(img)
        img = img * 255
        return np.int16(img)

    # def forward(self, Kspace):
    #
    #     rec_all_Kspace = Kspace
    #     if self.masked_kspace:
    #         rec_Kspace = self.mask*Kspace[:, int(Kspace.shape[1]/2)-1:int(Kspace.shape[1]/2)+1, :, :]
    #         rec_mid_img = self.inverseFT(rec_Kspace)
    #     else:
    #         rec_Kspace = rec_all_Kspace
    #         rec_mid_img = self.fftshift(self.inverseFT(rec_Kspace))
    #
    #     refine_Img = self.img_UNet(rec_mid_img)
    #
    #     rec_mid_img_np=rec_mid_img.cpu().detach().numpy()
    #     refine_Img_np=refine_Img.cpu().detach().numpy()
    #
    #     # rec_img = torch.tanh(refine_Img + rec_mid_img)
    #
    #     rec_img_np=refine_Img.cpu().detach().numpy()
    #
    #     rec_img = torch.clamp(refine_Img, 0, 1)
    #     # if self.train():
    #     return rec_img, rec_Kspace, rec_mid_img