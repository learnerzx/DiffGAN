import numpy
import numpy as np
import glob
import nibabel as nib
import os
import random
import h5py
import shutil
import pydicom
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pathlib
from data_process import subsample
from data_process import transforms, mri_data

# Download IXI T1 dataset from:  'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar'
# extract with:  tar -xvf IXI-T1.tar
# and save at "nii path"
# output of script will be save in "save path"

nii_path = 'data/h5_FastMRI/singlecoil_train/'
save_path = 'data/h5_FastMRI_rd/'
TEST_PERCENTAGE = 0.2
VAL_PERCENTAGE = 0.16

nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 注意是负号哦


def convert2hdf5(file_path):
    # try:
    # Create a mask function
    # read:
    # data = nib.load(file_path).get_data()
    # dcm = pydicom.dcmread(file_path)

    data = h5py.File(file_path, 'r')
    data = np.array(data['reconstruction_rss'])


    # Norm data:
    data = (data - data.min()) / (data.max() - data.min()).astype(np.float32)

    # data_split = []
    # for i in range(data.shape[2] // 2 - 40, data.shape[2] // 2 + 40):
    #     img_2d = data[:, :, i]
    #     data_split.append(img_2d)
    # data_split=numpy.array(data_split).transpose(1,2,0)

    # 数据可视化展示
    # plt.figure()
    # for i in range(data.shape[0]):
    #     plt.subplot(6, 6, i+1)  # 将画板分为2行两列，本幅图位于第3个位置
    #     plt.imshow(data[i])
    # plt.show()

    # save hdf5:
    data_shape = data.shape
    patient_name = os.path.split(file_path)[1].replace('h5', 'hdf5')
    output_file_path = save_path + patient_name
    with h5py.File(output_file_path, 'w') as f:
        dset = f.create_dataset('data', data_shape, data=data, compression="gzip", compression_opts=9)


    # patient_name = os.path.split(file_path)[1].replace('h5', 'hdf5')
    # output_file_path = save_path + patient_name
    # with h5py.File(output_file_path, 'w') as f:
    #     dset = f.create_dataset('data', data_shape, data=data, compression="gzip", compression_opts=9)


# except:
#     print(file_path, ' Error!')

def move_split(new_data_dir, source_data_list):
    os.makedirs(new_data_dir, exist_ok=True)
    for source_data in source_data_list:
        shutil.move(src=source_data, dst=new_data_dir)


if __name__ == '__main__':
    data_list = glob.glob(nii_path + '*.h5')
    os.makedirs(save_path, exist_ok=True)

    P = Pool(10)
    P.map(convert2hdf5, data_list)

    h5_list = glob.glob(save_path + '*.hdf5')

    num_files = len(h5_list)
    num_test = int(num_files * TEST_PERCENTAGE)
    num_val = int(num_files * VAL_PERCENTAGE)
    random.shuffle(h5_list)
    test_list = h5_list[:num_test]
    val_list = h5_list[num_test:(num_test + num_val)]
    train_list = h5_list[(num_test + num_val):]

    with open(save_path + 'split.txt', 'w') as f:
        f.writelines(['train:\n'])
        [f.writelines(os.path.split(t)[1] + '\n') for t in train_list]
        f.writelines(['\nval:\n'])

        [f.writelines(os.path.split(t)[1] + '\n') for t in val_list]
        f.writelines(['\ntest:\n'])

        [f.writelines(os.path.split(t)[1] + '\n') for t in test_list]

    move_split(save_path + 'train', train_list)
    move_split(save_path + 'val', val_list)
    move_split(save_path + 'test', test_list)
