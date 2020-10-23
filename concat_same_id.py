import cv2
import numpy as np
import os
import os.path as osp
import argparse
from tqdm import tqdm

def write_imgs(pid2img, dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

    border = 5

    height = 256

    for pid, img_list in tqdm(pid2img.items()):
        # img_cv_list = [cv2.copyMakeBorder(cv2.imread(img_path), border, border, border, border, cv2.BORDER_CONSTANT)
        #                for img_path in img_list]
        img_cv_list = []
        for img_path in (img_list):
            img = cv2.imread(img_path)
            img_h = img.shape[0]
            img_w = img.shape[1]
            width = int(img_w * height * 1.0 / img_h)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT)
            img_cv_list.append(img)
        img_all = np.concatenate(img_cv_list, axis=1)
        cv2.imwrite(osp.join(dir_path, '{}.png'.format(pid)), img_all)

def shorter_pid2img(pid2img, max_length):
    spid2img = {}
    for pid, img_list in pid2img.items():
        if len(img_list) <= max_length:
            spid2img[pid] = img_list
        else:
            spid2img[pid] = []
            interval = (len(img_list) / max_length) + 1
            for i, img in enumerate(img_list):
                if i % interval == 0:
                    spid2img[pid].append(img)
    return spid2img

def process_imgs(images_path, max_length):
    imgs_filenames = os.listdir(images_path)
    pid2img = {}
    for img_filename in imgs_filenames:
        id = int(img_filename.split('_')[0])
        if id in pid2img.keys():
            pid2img[id].append(osp.join(images_path, img_filename))
        else:
            pid2img[id] = []
            pid2img[id].append(osp.join(images_path, img_filename))

    for pid, img_list in pid2img.items():
        pid2img[pid] = sorted(img_list)

    if max_length is not None:
        pid2img = shorter_pid2img(pid2img, max_length)

    return pid2img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dir_path = args.save_path

    pid2img = process_imgs(args.images_path, args.max_length)

    write_imgs(pid2img, dir_path)