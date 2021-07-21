import os.path
from pathlib import Path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS



class VCIPNir2RGBDataset(BaseDataset):

 

    def __init__(self, opt):

 

        BaseDataset.__init__(self, opt)

 

        self.dir_A = Path(opt.dataroot, 'NIR')  # create a path '/path/to/data/trainA'

        self.dir_B_1 = Path(opt.dataroot, 'RGB-Registered')  # create a path '/path/to/data/trainB1'

        self.dir_B_2 = Path(opt.dataroot, 'RGB-Online') # create a path '/path/to/data/trainB2'

 

        self.A_paths= [f for f in self.dir_A.glob('*') if is_image(f)]

        self.B1_paths= [f for f in self.dir_B_1.glob('*') if is_image(f)]

        self.B1_pair_paths= [f for f in self.dir_B_1.glob('*') if is_image(f)]

        self.B2_paths= [f for f in self.dir_B_2.glob('*') if is_image(f)]

 

        self.A_paths.sort()

        self.B1_pair_paths.sort()

 

        self.A_size = len(self.A_paths)  # get the size of dataset A

        self.B_size = len(self.B1_paths) + len(self.B2_paths)  # get the size of dataset B

 

    def __getitem__(self, index):

      

        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        B_pair_RGB = self.B1_pair_paths[index % self.A_size]

 

        if self.opt.serial_batches:   # make sure index is within the range

            index_B = index % self.B_size

        else:   # randomize the index for domain B to avoid fixed pairs.

            index_B = random.randint(0, self.B_size - 1)

 

        if index_B < len(self.B1_paths):

            B_path = self.B1_paths[index_B]

        else:

            B_path = self.B2_paths[index_B-len(self.B1_paths)]

 

        #print(B_path)   

        #B_path = self.B_paths[index_B]

        A_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1))/255

        B_img = np.float32(np.asarray(Image.open(B_path).convert('RGB')).transpose(2,0,1))/255


        # Image.fromarray(np.uint8(255*A_img.transpose(1,2,0))).show()

       

        # return {'A': A_img[0:1], 'B': B_img[0:self.opt.output_nc],'B_gray':B_gray_img[0:1],'B_RGB':B_pair_RGB_img[0:self.opt.input_nc],'A_paths': str(A_path), 'B_paths': str(B_path)}
        return {'A': A_img[0:self.opt.input_nc], 'B': B_img[0:self.opt.output_nc], 'A_paths': str(A_path), 'B_paths': str(B_path)}

 

    def __len__(self):

        """Return the total number of images in the dataset.

 

        As we have two datasets with potentially different number of images,

        we take a maximum of

        """

        return max(self.A_size, self.B_size)

class VCIPNir2RGBDataset_test(BaseDataset):
 
    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.dir_A = Path(opt.dataroot)  # create a path '/path/to/data/trainA'
        self.dir_B = Path(opt.dataroot)  # create a path '/path/to/data/trainB'

        self.A_paths= [f for f in self.dir_A.glob('*_nir.png') if is_image(f)]
        self.B_paths= [f for f in self.dir_B.glob('*_rgb_reg.png') if is_image(f)]

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
      
        A_path = self.A_paths[index]  
        B_path = self.B_paths[index]

        A_img = np.float32(np.asarray(Image.open(A_path).convert('RGB')).transpose(2,0,1))/255
        B_img = np.float32(np.asarray(Image.open(B_path).convert('RGB')).transpose(2,0,1))/255
        # Image.fromarray(np.uint8(255*A_img.transpose(1,2,0))).show()
       
        return {'A': A_img, 'B': B_img, 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)