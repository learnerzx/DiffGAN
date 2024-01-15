import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def subtract(src,tgt):
    result=cv.subtract(src,tgt)
    # result=result.astype(np.int32)

    # im_gray = cv.imread("pluto.jpg", cv.IMREAD_GRAYSCALE)

    sc = plt.imshow(result, cmap=plt.cm.jet)  # 设置cmap为RGB图

    # cmap = matplotlib.cm.jet
    # ax = [0.3, 0.2, 0.2, 0.5]
    # norm = matplotlib.colors.Normalize(vmin=1.3, vmax=2.5)
    # Normalizer
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)                              ## rgb color bar
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.gray, norm=norm)                              ## gray color bar

    cb=plt.colorbar(sm, ticks=np.linspace(0, 1, num=11, retstep=False), format='%.1f')

    cb.ax.tick_params(labelsize=20)  # 设置色标刻度字体大小。


    # plt.colorbar(mappable=None,norm=norm)  # 显示色度条

    plt.show()
    # sc.show()


    # im_color = cv.applyColorMap(result, cv.COLORMAP_JET)-9
    # cv.imshow('error_map',im_color)
    # cv.waitKey(0)

if __name__ == '__main__':
    root=r"E:\code\code_backup_vvt+diffusion\SAVE_path"
    # root='C:/Users/zx/Desktop/cartesian_30%'
    src = root+'/3_K_rec.png'
    tgt = root+'/3_gd.png'

    # src = np.load(src, 'r')
    # tgt = np.load(tgt, 'r')

    src1 =cv.imread(src, cv.IMREAD_GRAYSCALE)
    tgt1 =cv.imread(tgt, cv.IMREAD_GRAYSCALE)
    subtract(src1,tgt1)
