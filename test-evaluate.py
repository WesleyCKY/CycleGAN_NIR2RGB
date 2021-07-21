import numpy as np
from skimage.metrics import structural_similarity as ssim
import glob, os
from PIL import Image

GT_path = "C:/Users/wesle/fyp/CycleGAN_NIR_clean/data/NIR_VCIP_Challenge_dataset/Validation"
GT_file = os.listdir(GT_path)
GT_file = [f for f in GT_file if f.endswith('.png')]
GT_file =[f for f in GT_file if f.endswith('_rgb_reg.png') and f.startswith("validation_")]
GT_file.sort()

im_path = "C:/Users/wesle/fyp/Experiments/CycleGAN-NIR_NI2RRGB_ResUnet_Test2/checkpoints/result/NI2RRGB_ResUnet_Test2/test_latest/images/"
im_file = os.listdir(im_path)
im_file = [f for f in im_file if f.endswith('.png')]
im_file =[f for f in im_file if f.endswith('fake_A.png') or f.endswith('fake_B.png') ]
im_file.sort()

PSNR_list = []


SSIM_list = []

AE_list = []

# len(GT_file)

for i in range(0, len(GT_file)):

    GT = Image.open(os.path.join(GT_path,GT_file[i])).convert('RGB')

    im = Image.open(os.path.join(im_path,im_file[i])).convert('RGB')

    GT_arr = np.float32(np.asarray(GT))/255

    im_arr = np.float32(np.asarray(im))/255

    PSNR = -10*np.log10(np.mean((im_arr - GT_arr)**2))

    PSNR_list.append(PSNR)

##part b
    SSIM = ssim(GT_arr, im_arr, data_range=im_arr.max() - im_arr.min(), multichannel=True)
 
    SSIM_list.append(SSIM)

    eps = 1e-6

    dotP = np.sum(GT_arr * im_arr, axis = 2)

    Norm_pred = np.sqrt(np.sum(im_arr * im_arr, axis = 2))

    Norm_true = np.sqrt(np.sum(GT_arr * GT_arr, axis = 2))

    AE = 180 / np.pi * np.arccos(dotP / (Norm_pred * Norm_true + eps))

    AE = AE.ravel().mean()

    AE_list.append(AE)

print("SSIM list", SSIM_list)
print("AE list",AE_list)
print("PSNR list",PSNR_list)


output = open('data validation_Proposed_method.xls','w',encoding='gbk')
output.write('Ground true path:\t'+GT_path+'\n'+ 'generated image path:\t' + im_path +'\n')

output.write('SSIM list\t') #结构相似性 - Structural Similarity Index 
                            #两幅图像相似度的指标 当两张图像一模一样时，SSIM的值等于1
for i in range(len(SSIM_list)):   
    output.write(str(SSIM_list[i]))
    output.write('\t')
output.write('\n')  

output.write('AE list\t') #平均绝对误差

for i in range(len(AE_list)):
    output.write(str(AE_list[i]))
    output.write('\t')
output.write('\n')  

output.write('PSNR list\t')  #Peak signal-to-noise ratio  #图像压缩中典型的峰值信噪比值在 30 到 40dB 之间，越高越好。
                             #https://www.cnblogs.com/tiandsp/archive/2012/11/14/2770462.html
for i in range(len(PSNR_list)):   
    output.write(str(PSNR_list[i]))
    output.write('\t')
output.write('\n')  

output.close()
print('file generated')