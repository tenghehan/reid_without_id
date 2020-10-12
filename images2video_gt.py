import os.path as osp
import os
import cv2
import argparse
from tqdm import tqdm

from numpy.core.numeric import identity

from utils.log import get_logger
from utils.draw import draw_boxes

def process_gt(gt_filepath):
    gt = []
    for line in open(gt_filepath):
        info = line.split(',')
        idx_frame = int(info[0])
        identity = int(info[1])
        # bbox: tlwh
        bbox = (int(info[2]), int(info[3]), int(info[4]), int(info[5]))
        consider = int(info[6])
        info_dict = {
            'idx_frame': idx_frame,
            'identity': identity,
            'bbox': bbox
        }
        if consider:
            gt.append(info_dict)
    return gt

def generate_video(gt, imgs_filenames, writer):
    idx_frame = 0
    for img_filename in tqdm(imgs_filenames):
        idx_frame += 1
        ori_im = cv2.imread(os.path.join(args.IMAGES_PATH, 'img1', img_filename))
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

        bbox_xyxy = []
        identities = []
        for track in gt:
            if track["idx_frame"] == idx_frame:
                bbox_xyxy.append((track["bbox"][0], track["bbox"][1], track["bbox"][0] + track["bbox"][2], track["bbox"][1] + track["bbox"][3]))
                identities.append(track["identity"])
        ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
        writer.write(ori_im)


def run(imgs_filenames, gt_filepath):
    first_img = cv2.imread(osp.join(args.IMAGES_PATH, 'img1', imgs_filenames[0]))
    im_width = first_img.shape[1]
    im_height = first_img.shape[0]

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)

        # path of saved video
        filename = osp.split(args.IMAGES_PATH)[-1]
        assert filename != '', "Filename error"
        save_video_path = osp.join(args.save_path, f'{filename}.avi')

        # create video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(save_video_path, fourcc, args.fps, (im_width, im_height))

        # logging
        logger = get_logger("root")
        logger.info("Save results to {}".format(save_video_path))

        gt = process_gt(gt_filepath)
        generate_video(gt, imgs_filenames, writer)

def main():
    assert osp.isdir(args.IMAGES_PATH), "Path error"
    assert osp.isdir(osp.join(args.IMAGES_PATH, "img1")), "Image sequence can not be found"
    assert osp.isdir(osp.join(args.IMAGES_PATH, "gt")), "Ground truth file can not be found"
    imgs_filenames = sorted(os.listdir(osp.join(args.IMAGES_PATH, "img1")))
    gt_filepath = osp.join(args.IMAGES_PATH, "gt/gt.txt")
    
    run(imgs_filenames, gt_filepath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("IMAGES_PATH", type=str)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="./videos_gt/")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main()