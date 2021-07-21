
import os
from options.test_options import TestOptions

from data.VCIP_nir2rgb_dataset import *
from models.CycleGanNIR_model import *

from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.

    dataset= VCIPNir2RGBDataset_test(opt) # create dataset
    print("dataset [%s] was created" % type(dataset).__name__)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, 
                    shuffle=not opt.serial_batches, num_workers=int(opt.num_threads))
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of testing images = %d' % dataset_size)

    model = CycleGANModel(opt)       # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataloader):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        # PSNR SSIM AE

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
    webpage.save()  # save the HTML
