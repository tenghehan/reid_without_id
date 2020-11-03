"""
@author: tenghehan

将 MOT video 的前部分利用 tracking result 生成 reid train dataset.
"""
import os
import cv2
import argparse
import random
import shutil
from numpy.lib.function_base import delete

from tqdm import tqdm
from utils.log import get_logger
from utils.txt_logger import txt_logger

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


class ReIDTrainDataConverter():
    def __init__(self, args):
        self.logger = get_logger('root')
        self.txt_logger = txt_logger(os.path.join(args.save_path, args.dataset_name, 'info.txt'))

        self.dataset_name = args.dataset_name
        self.images_path = os.path.join(args.image_sequence_path, self.dataset_name, "img1")
        assert os.path.isdir(self.images_path), "Images path error"
        self.imgs_filenames = os.listdir(os.path.join(args.image_sequence_path, self.dataset_name, "img1"))
        self.frame_length = len(self.imgs_filenames)

        self.track_file_path = os.path.join(args.track_result_path, self.dataset_name + ".txt")
        assert os.path.isfile(self.track_file_path), "track file path error"

        self.save_path = os.path.join(args.save_path, self.dataset_name)

        os.makedirs(os.path.join(self.save_path, 'train'), exist_ok=True)

        self.frame_interval = args.frame_interval
        self.train_rate = args.train_rate
        self.end_frame = int(self.frame_length * self.train_rate) - 1

        first_img = cv2.imread(os.path.join(self.images_path, self.imgs_filenames[0]))
        self.im_width = first_img.shape[1]
        self.im_height = first_img.shape[0]


    def process_gt_result(self):
        self.train_set_details = []
        for line in open(self.track_file_path):
            info = line.split(',')
            idx_frame = int(info[0])
            identity = int(info[1])
            if idx_frame > self.end_frame:
                continue

            # bbox: tlwh
            bbox = (int(info[2]), int(info[3]), int(info[4]),int(info[5]))
            info_dict = {
                'idx_frame': idx_frame,
                'identity': identity,
                'bbox': bbox
            }
            self.train_set_details.append(info_dict)


    def generate_train_dataset(self):
        self.trainset_size = 0
        self.train_id = set()
        for info in tqdm(self.train_set_details):
            idx_frame = info['idx_frame']
            if idx_frame % self.frame_interval:
                continue

            
            # read frame image
            frame_path = os.path.join(self.images_path, f'{str(idx_frame).zfill(6)}.jpg')
            frame = cv2.imread(frame_path)

            # crop the person area from the whole image
            x1 = max(info['bbox'][0], 0)
            y1 = max(info['bbox'][1], 0)
            x2 = min(info['bbox'][0] + info['bbox'][2], self.im_width)
            y2 = min(info['bbox'][1] + info['bbox'][3], self.im_height)
            if x2 <= x1 or y2 <= y1:
                continue

            self.train_id.add(info['identity'])
            cropped_img = frame[y1:y2, x1:x2]
            img_name = str(info['identity']).zfill(5) + '_c1_' + str(idx_frame).zfill(6) + '.jpg'
            self.trainset_size += 1

            cv2.imwrite(os.path.join(self.save_path, 'train', img_name), cropped_img)


    def run(self):
        self.txt_logger.add_info('for track-train set:')
        self.txt_logger.add_info('frame interval: {}'.format(self.frame_interval))
        self.txt_logger.add_info('train rate: {}'.format(self.train_rate))

        self.process_gt_result()
        
        self.generate_train_dataset()

        self.txt_logger.add_info('trainset: {} ids, {} images'.format(
            len(self.train_id), self.trainset_size
        ))

        self.txt_logger.output_tail()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_sequence_path", type=str, default="./image_sequence/")
    parser.add_argument("--save_path", type=str, default="./reid_dataset/")
    parser.add_argument("--track_result_path", type=str, default="./output/")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--train_rate", type=float, default=0.5, choices=[Range(0.0, 1.0)], help="percentage (frames) of train set")

    return parser.parse_args()


if __name__ == "__main__":
    
    rand_seed = 50
    random.seed(rand_seed)

    args = parse_args()

    reidTrainDataConverter = ReIDTrainDataConverter(args)
    reidTrainDataConverter.run()
