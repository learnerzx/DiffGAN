from Masks.mask_generator import Gaussian2DMaskFunc, RadialMaskFunc, SpiralMaskFunc,VariableDensityPoissonMaskFunc,FastMRIRandomMaskFunc
import os
import logging
import shutil
import scipy.io as scio
mask_type='Radial'
accelerations=[4]
center_fractions=[0.08]
seed=10

if __name__ == '__main__':

    mask_types=['Gaussian2D','Radial','Spiral','Poisson','Random']
    accelerationsS=[[2],[4],[6],[8]]
    root='E:/code/code_backup_vvt+diffusion/Masks_new/'

    for mask_type in mask_types:
        for accelerations in accelerationsS:
            if mask_type == 'Gaussian2D':

                mask_func = Gaussian2DMaskFunc(accelerations=accelerations, center_fractions=center_fractions)
                mask = mask_func([256, 256, 1], seed=seed).permute(3, 0, 1, 2)
                mask = mask.numpy()
                save_path=root+'%s/' % mask_type
                os.makedirs(save_path, exist_ok=True)
                scio.savemat(os.path.join(save_path,'%s.mat'% accelerations),{'mask1': mask})

            elif mask_type == 'Radial':

                mask_func = RadialMaskFunc(accelerations=accelerations, center_fractions=center_fractions)
                mask = mask_func([256, 256, 1], seed=seed).permute(3, 0, 1, 2)
                mask = mask.numpy()
                save_path=root+'%s/' % mask_type
                os.makedirs(save_path, exist_ok=True)
                scio.savemat(os.path.join(save_path,'%s.mat'% accelerations),{'mask1': mask})

            elif mask_type == 'Spiral':

                mask_func = SpiralMaskFunc(accelerations=accelerations, center_fractions=center_fractions)
                mask = mask_func([256, 256, 1], seed=seed).permute(3, 0, 1, 2)
                mask = mask.numpy()
                save_path=root+'%s/' % mask_type
                os.makedirs(save_path, exist_ok=True)
                scio.savemat(os.path.join(save_path,'%s.mat'% accelerations),{'mask1': mask})

            elif mask_type == 'Poisson':

                mask_func = VariableDensityPoissonMaskFunc(accelerations=accelerations, center_fractions=center_fractions)
                mask = mask_func([256, 256, 1], seed=seed).permute(3, 0, 1, 2)
                mask = mask.numpy()
                save_path=root+'%s/' % mask_type
                os.makedirs(save_path, exist_ok=True)
                scio.savemat(os.path.join(save_path,'%s.mat'% accelerations),{'mask1': mask})

            elif mask_type == 'Random':

                mask_func = FastMRIRandomMaskFunc(accelerations=accelerations, center_fractions=center_fractions)
                mask = mask_func([256, 256, 1], seed=seed).permute(3, 0, 1, 2)
                mask = mask.numpy()
                save_path = root + '%s/' % mask_type
                os.makedirs(save_path, exist_ok=True)
                scio.savemat(os.path.join(save_path, '%s.mat' % accelerations), {'mask1': mask})

            else:
                raise ValueError('Unknown mask type {}'.format(mask_type))










    # mask=torch.cat([mask,mask],dim=1).cuda()
    # maskNot = mask == 0
    # maskNot = maskNot.cuda()

