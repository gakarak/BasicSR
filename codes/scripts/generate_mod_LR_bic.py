import os
import sys
import cv2
import numpy as np
import argparse
import pandas as pd

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


def generate_mod_LR_bic(path_idx: str, scale: int):
    # set parameters
    up_scale = scale
    mod_scale = scale
    # set data dir
    wdir = os.path.dirname(path_idx)
    data_idx = pd.read_csv(path_idx)
    paths_img = [os.path.join(wdir, x) for x in data_idx['path']]

    # sourcedir = '/data/datasets/img'
    savedir = path_idx + '_mod'
    os.makedirs(savedir, exist_ok=True)

    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(mod_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    # if not os.path.isdir(sourcedir):
    #     print('Error: No source data found')
    #     exit(0)
    # if not os.path.isdir(savedir):
    #     os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))

    num_files = len(paths_img)
    # prepare data with augementation
    for i, path_img in enumerate(paths_img):
        filename = os.path.basename(path_img)
        print('No.{} -- Processing {}'.format(i, filename))
        # read image
        image = cv2.imread(path_img)
        width = int(np.floor(image.shape[1] / mod_scale))
        height = int(np.floor(image.shape[0] / mod_scale))
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
        else:
            image_HR = image[0:mod_scale * height, 0:mod_scale * width]
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        # bic
        image_Bic = imresize_np(image_LR, up_scale, True)
        cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)
        # if (i % 10) ==0:
        #     print('\t[{}/{}] <- ({})'.format(i, num_files, path_img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default=None, required=True, help='Index file with HR images')
    parser.add_argument('--scale', type=int, default=4, required=False, help='Scale factor for LR image generation')
    args = parser.parse_args()
    print('args:\n\t{}'.format(args))
    #
    generate_mod_LR_bic(
        path_idx=args.idx,
        scale=args.scale
    )
