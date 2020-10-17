"""
@author: tenghehan

将 video 形式的视频数据转化成 image sequence 形式的数据.
"""
import cv2
import os
import argparse
from tqdm import tqdm

from utils.log import get_logger

class VideoFramer(object):
    def __init__(self, args, video_path):
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        self.vdo = cv2.VideoCapture()

    def __enter__(self):
        assert os.path.isfile(self.video_path), "Path error"
        self.vdo.open(self.video_path)
        self.total_frames = int(cv2.VideoCapture.get(self.vdo, cv2.CAP_PROP_FRAME_COUNT))
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()

        if self.args.save_path:

            # path of saved image sequence
            dirname, _ = os.path.splitext(os.path.basename(self.video_path))
            self.args.save_path = os.path.join(self.args.save_path, dirname, "img1")

            os.makedirs(self.args.save_path, exist_ok=True)

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        pbar = tqdm(self.total_frames + 1)
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            _, ori_im = self.vdo.retrieve()

            image_save_path = os.path.join(self.args.save_path, f'{str(idx_frame).zfill(6)}.jpg')

            cv2.imwrite(image_save_path, ori_im)

            pbar.update()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--save_path", type=str, default="./image_sequence/")
    parser.add_argument("--frame_interval", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with VideoFramer(args, video_path=args.VIDEO_PATH) as vdo_frm:
        vdo_frm.run()