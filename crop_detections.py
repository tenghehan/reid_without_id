"""
@author: tenghehan

对 image sequence 作 detection，并将 detection 结果裁切出来存储在指定文件夹中.
"""
import os
import cv2
import argparse
import torch
import warnings

from detector import build_detector
from utils.parser import get_config
from utils.log import get_logger
from tqdm import tqdm

def xywh_to_xyxy(boxes_xywh):
    boxes_xyxy = boxes_xywh.copy()
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.

    return boxes_xyxy

class DetectionCropper(object):
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
            dataset_name = os.path.split(self.images_path)[-1]
            assert dataset_name != '', "Filename error"
            self.save_path = self.args.save_path
            self.dataset_name = dataset_name

            # logging
            self.logger.info("Save results to {}".format(self.save_path))
        
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
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

            bbox_xyxy = xywh_to_xyxy(bbox_xywh)

            for i, bb_xyxy in enumerate(bbox_xyxy):
                x1 = max(int(bb_xyxy[0]), 0)
                y1 = max(int(bb_xyxy[1]), 0)
                x2 = min(int(bb_xyxy[2]), self.im_width)
                y2 = min(int(bb_xyxy[3]), self.im_height)
                if x2 <= x1 or y2 <= y1:
                    continue
                cropped_img = ori_im[y1:y2, x1:x2]
                img_name = str(self.dataset_name) + '_' + str(idx_frame).zfill(6) + '_' + str(i) + '.jpg'

                cv2.imwrite(os.path.join(self.save_path, img_name), cropped_img)

  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("IMAGES_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--save_path", type=str, default="./detections/")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)

    with DetectionCropper(cfg, args, images_path=args.IMAGES_PATH) as det_cropper:
        det_cropper.run()

