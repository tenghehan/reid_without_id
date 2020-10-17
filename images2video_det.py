"""
@author: tenghehan

对 image sequence 作 detection，并将 detection 结果绘制在图片序列上，并合成视频.
"""
import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from PIL import Image
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from tqdm import tqdm

def xywh_to_xyxy(boxes_xywh):
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.

    return boxes_xyxy

class ImageSequenceDetector(object):
    def __init__(self, cfg, args, images_path):
        self.cfg = cfg
        self.args = args
        self.images_path = images_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

       
        assert os.path.isdir(self.images_path), "Path error"
        self.imgs_filenames = sorted(os.listdir(os.path.join(self.images_path, 'img1')))
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        first_img = cv2.imread(os.path.join(self.images_path, 'img1', self.imgs_filenames[0]))
        self.im_width = first_img.shape[1]
        self.im_height = first_img.shape[0]


        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            filename = os.path.split(self.images_path)[-1]
            assert filename != '', "Filename error"
            self.save_video_path = os.path.join(self.args.save_path, f'{filename}.avi')
            self.save_results_path = os.path.join(self.args.save_path, f'{filename}.txt')
            self.save_json_path = os.path.join(self.args.save_path, f'{filename}.json')

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, self.args.fps, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        for img_filename in tqdm(self.imgs_filenames):
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            ori_im = cv2.imread(os.path.join(self.images_path, 'img1', img_filename))
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # draw boxes for visualization
            ori_im = draw_boxes(ori_im, xywh_to_xyxy(bbox_xywh), cls_conf)

            if self.args.save_path:
                self.writer.write(ori_im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("IMAGES_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)

    with ImageSequenceDetector(cfg, args, images_path=args.IMAGES_PATH) as imgs_det:
        imgs_det.run()
