import numpy as np
from PIL import Image
import h5py
import matplotlib.pyplot as plt
import tqdm

# List images and label them
from random import shuffle
from pathlib import Path
from PIL import Image

import torch.nn.functional as F

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.TIFF',
]

cropSize= 680
finalSize= 256

# process hdf5 files
epfl_dataset_path= 'D:/Datasets/NIR_VCIP/Srouce/EPFL/'
cat= ['country', 'field', 'forest', 'mountain']
# lf_data_list= [f for f in curr_path.iterdir() if f.is_dir()]
epfl_data_list= [Path(epfl_dataset_path + cat[n] + '/') for n in range(len(cat))]

validation_dataset_path= 'D:/Datasets/NIR_VCIP/Srouce/EPFL/'
cat= ['validation']
# lf_data_list= [f for f in curr_path.iterdir() if f.is_dir()]
validation_data_list= [Path(epfl_dataset_path + cat[n] + '/') for n in range(len(cat))]

onl_dataset_path= 'D:/Datasets/NIR_VCIP/Srouce/ONL/'
cat= ['country', 'field', 'forest', 'mountain']
onl_data_list= [Path(onl_dataset_path + cat[n] + '/') for n in range(len(cat))]

lf_savepath= 'D:/Datasets/NIR_VCIP/'


def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS

def EPFL_dataset_from_folder(addr_list):

    # hdf5_path = lf_dataset_path + f'InriaSyn_dataset.hdf5'  # address to where you want to save the hdf5 file
    # hdf5_file = h5py.File(hdf5_path, mode='w')

    for idx_cat in tqdm.tqdm(range(len(addr_list))):

        # img_list= [f for f in addr_list[idx_cat].glob('*.tiff') if is_image(f)]
        img_list_rgb= [f for f in addr_list[idx_cat].glob('*_rgb.tiff')]
        img_list_nir= [f for f in addr_list[idx_cat].glob('*_nir.tiff')]

        localIdx = np.random.permutation(2 * len(img_list_rgb))
        localCounter= 0

        catname= addr_list[idx_cat].stem.split('.')[0]

        for idx in range(len(img_list_rgb)):

            filename_base= img_list_rgb[idx].stem.split('_rgb')[0]
            

            imrgb = Image.open(img_list_rgb[idx]).convert('RGB')
            imnir = Image.open(img_list_nir[idx])
            W, H = imrgb.size
            
            currCropSize= min(cropSize, min(H, W))
            rgb_a= imrgb.crop((0, 0, min(currCropSize, W), min(currCropSize, H))).resize((finalSize, finalSize), resample = Image.BICUBIC)
            rgb_b= imrgb.crop((max(0, W- currCropSize), max(0, H- currCropSize), W, H)).resize((finalSize, finalSize), resample = Image.BICUBIC)

            nir_a= imnir.crop((0, 0, min(currCropSize, W), min(currCropSize, H))).resize((finalSize, finalSize), resample = Image.BICUBIC)
            nir_b= imnir.crop((max(0, W- currCropSize), max(0, H- currCropSize), W, H)).resize((finalSize, finalSize), resample = Image.BICUBIC)        

            if not Path(lf_savepath).exists():
                Path(lf_savepath).mkdir()

            lf_nirpath= Path(lf_savepath + '/NIR/')
            if not lf_nirpath.exists():
                Path(lf_nirpath).mkdir()

            lf_rgbpath= Path(lf_savepath + '/RGB-Registered/')
            if not lf_rgbpath.exists():
                Path(lf_rgbpath).mkdir()
            
            filename= f'{localIdx[localCounter]:04d}'
            localCounter += 1

            rgb_a.save(str(lf_rgbpath) + '/' + catname + '_' + filename + '_rgb_reg.png', format='png')
            nir_a.save(str(lf_nirpath) + '/' + catname + '_' + filename + '_nir.png', format='png')

            filename= f'{localIdx[localCounter]:04d}'
            localCounter += 1

            rgb_b.save(str(lf_rgbpath) + '/' + catname + '_' + filename + '_rgb_reg.png', format='png')            
            nir_b.save(str(lf_nirpath) + '/' + catname + '_' + filename + '_nir.png', format='png')

        '''
        hdf5_file.create_dataset(filename + "/SV", (angDim[0], angDim[1], H, W, 3), np.uint8)
        hdf5_file.create_dataset(filename + "/disparity", (H, W), np.float32)
        # hdf5_file.create_dataset(filename + "/depth", (H, W), np.float32)

        hdf5_file[filename + "/SV"][...]= SV
        hdf5_file[filename + "/disparity"][...]= disparity_map
        # hdf5_file[filename + "/depth"][...]= depth_map'''

    # hdf5_file.close()

def EPFL_Validation_dataset_from_folder(addr_list):

    idx_cat= 0
    # img_list= [f for f in addr_list[idx_cat].glob('*.tiff') if is_image(f)]
    img_list_rgb= [f for f in addr_list[idx_cat].glob('*_rgb.tiff')]
    img_list_nir= [f for f in addr_list[idx_cat].glob('*_nir.tiff')]

    localIdx = np.random.permutation(2 * len(img_list_rgb))
    localCounter= 0

    catname= addr_list[idx_cat].stem.split('.')[0]

    for idx in range(len(img_list_rgb)):

        filename_base= img_list_rgb[idx].stem.split('_rgb')[0]
        

        imrgb = Image.open(img_list_rgb[idx]).convert('RGB')
        imnir = Image.open(img_list_nir[idx])
        W, H = imrgb.size
        
        currCropSize= min(cropSize, min(H, W))
        rgb_a= imrgb.crop((0, 0, min(currCropSize, W), min(currCropSize, H))).resize((finalSize, finalSize), resample = Image.BICUBIC)
        rgb_b= imrgb.crop((max(0, W- currCropSize), max(0, H- currCropSize), W, H)).resize((finalSize, finalSize), resample = Image.BICUBIC)

        nir_a= imnir.crop((0, 0, min(currCropSize, W), min(currCropSize, H))).resize((finalSize, finalSize), resample = Image.BICUBIC)
        nir_b= imnir.crop((max(0, W- currCropSize), max(0, H- currCropSize), W, H)).resize((finalSize, finalSize), resample = Image.BICUBIC)        

        if not Path(lf_savepath).exists():
            Path(lf_savepath).mkdir()

        lf_nirpath= Path(lf_savepath + '/Validation/')
        if not lf_nirpath.exists():
            Path(lf_nirpath).mkdir()

        lf_rgbpath= Path(lf_savepath + '/Validation/')
        if not lf_rgbpath.exists():
            Path(lf_rgbpath).mkdir()
        
        filename= f'{localIdx[localCounter]:04d}'
        localCounter += 1

        rgb_a.save(str(lf_rgbpath) + '/' + catname + '_' + filename + '_rgb_reg.png', format='png')
        nir_a.save(str(lf_nirpath) + '/' + catname + '_' + filename + '_nir.png', format='png')

        filename= f'{localIdx[localCounter]:04d}'
        localCounter += 1

        rgb_b.save(str(lf_rgbpath) + '/' + catname + '_' + filename + '_rgb_reg.png', format='png')            
        nir_b.save(str(lf_nirpath) + '/' + catname + '_' + filename + '_nir.png', format='png')



def ONL_dataset_from_folder(addr_list):

    # hdf5_path = lf_dataset_path + f'InriaSyn_dataset.hdf5'  # address to where you want to save the hdf5 file
    # hdf5_file = h5py.File(hdf5_path, mode='w')

    for idx_cat in tqdm.tqdm(range(len(addr_list))):

        # img_list= [f for f in addr_list[idx_cat].glob('*.tiff') if is_image(f)]
        img_list_rgb= [f for f in addr_list[idx_cat].glob('*')]

        localIdx = np.random.permutation(2 * len(img_list_rgb))
        localCounter= 0

        catname= addr_list[idx_cat].stem.split('.')[0]

        for idx in range(len(img_list_rgb)):

            filename_base= img_list_rgb[idx].stem.split('_')[1]
            

            imrgb = Image.open(img_list_rgb[idx]).convert('RGB')
            W, H = imrgb.size
            
            currCropSize= min(cropSize, min(H, W))
            rgb_a= imrgb.crop((0, 0, min(currCropSize, W), min(currCropSize, H))).resize((finalSize, finalSize), resample = Image.BICUBIC)
            rgb_b= imrgb.crop((max(0, W- currCropSize), max(0, H- currCropSize), W, H)).resize((finalSize, finalSize), resample = Image.BICUBIC)

            if not Path(lf_savepath).exists():
                Path(lf_savepath).mkdir()

            lf_rgbpath= Path(lf_savepath + '/RGB-Online/')
            if not lf_rgbpath.exists():
                Path(lf_rgbpath).mkdir()

            filename= f'{localIdx[localCounter]:04d}'
            localCounter += 1

            rgb_a.save(str(lf_rgbpath) + '/' + catname + '_' + '0' + filename + '_rgb_onl.png', format='png')

            filename= f'{localIdx[localCounter]:04d}'
            localCounter += 1

            rgb_b.save(str(lf_rgbpath) + '/' + catname + '_' + '0' + filename + '_rgb_onl.png', format='png')


ONL_dataset_from_folder(onl_data_list)

EPFL_dataset_from_folder(epfl_data_list)

EPFL_Validation_dataset_from_folder(validation_data_list)
