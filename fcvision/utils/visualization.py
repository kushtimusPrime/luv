import numpy as np
import matplotlib.pyplot as plt

def get_mask_vis(rgb_img,mask,channel=1,strength=2):
    '''
    make a nicely overlaid image of rgb and mask on top
    '''
    original_type=rgb_img.dtype
    rgb_img=rgb_img.copy().astype(np.float32)
    rgb_img[mask>0,channel]=(rgb_img[mask>0,channel]+strength*255)/(strength+1)
    return rgb_img.astype(original_type)

if __name__=='__main__':

    from fcvision.cameras.zed import ZedImageCapture
    zed = ZedImageCapture()
    iml,_=zed.capture_image()
    mask =np.ones((iml.shape[0],iml.shape[1]))
    vis=get_mask_vis(iml,mask)
    plt.imshow(vis)
    plt.show()