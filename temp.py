import h5py
from os.path import splitext
from os import listdir,path


data_dir=r'E:\code\code_backup2_MRNet\data\h5\train'
file_names = [splitext(file)[0] for file in listdir(data_dir)]



total=[]
for file_name in file_names:
    full_file_path = path.join(data_dir, file_name + '.hdf5')
    with h5py.File(full_file_path, 'r') as f:
        numOfSlice = f['data'].shape[2]
        shape=f['data'].shape
    total.append(numOfSlice)
    print(file_name+'.hdf5: '+str(numOfSlice)+'  shape: '+str(shape))
total.sort()
print(total)