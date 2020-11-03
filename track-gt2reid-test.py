"""
@author: tenghehan

将 MOT video 的后部分利用 ground truth 生成 correct reid test dataset.

注意剔除其中在前部分出现过的 id (train set 和 test set 中不能有相同 id 的人)
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


class ReIDTestDataConverter():
    def __init__(self, args):
        self.logger = get_logger('root')
        self.txt_logger = txt_logger(os.path.join(args.save_path, args.dataset_name, 'info.txt'))

        self.dataset_name = args.dataset_name
        self.images_path = os.path.join(args.image_sequence_path, self.dataset_name, "img1")
        assert os.path.isdir(self.images_path), "Images path error"
        self.imgs_filenames = os.listdir(os.path.join(args.image_sequence_path, self.dataset_name, "img1"))
        self.frame_length = len(self.imgs_filenames)

        self.gt_file_path = os.path.join(args.image_sequence_path, self.dataset_name, "gt/gt.txt")
        assert os.path.isfile(self.gt_file_path), "gt file path error"

        self.save_path = os.path.join(args.save_path, self.dataset_name)
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(os.path.join(self.save_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'query'), exist_ok=True)

        self.frame_interval = args.frame_interval
        self.test_rate = args.test_rate
        self.start_frame = int(self.frame_length * (1 - self.test_rate))

        first_img = cv2.imread(os.path.join(self.images_path, self.imgs_filenames[0]))
        self.im_width = first_img.shape[1]
        self.im_height = first_img.shape[0]


    def process_gt_result(self):
        self.train_id = set()
        self.test_set_details = []
        for line in open(self.gt_file_path):
            info = line.split(',')
            idx_frame = int(info[0])
            identity = int(info[1])
            if idx_frame < self.start_frame:
                self.train_id.add(identity)

        for line in open(self.gt_file_path):
            info = line.split(',')
            idx_frame = int(info[0])
            identity = int(info[1])
            if idx_frame < self.start_frame:
                continue
            if self.dataset_name in delete_ids.keys() and identity in delete_ids[self.dataset_name]:
                continue

            # bbox: tlwh
            bbox = (int(info[2]), int(info[3]), int(info[4]),int(info[5]))
            consider = int(info[6])
            type = int(info[7])
            visiblity = float(info[8])
            info_dict = {
                'idx_frame': idx_frame,
                'identity': identity,
                'bbox': bbox
            }
            if consider == 1 and type == 1 and visiblity >= 0.6 and identity not in self.train_id:
                self.test_set_details.append(info_dict)


    def generate_test_dataset(self):
        self.testset_size = 0
        self.id_images_details = {}
        for info in tqdm(self.test_set_details):
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
            cropped_img = frame[y1:y2, x1:x2]
            img_name = str(info['identity']).zfill(5) + '_c1_' + str(idx_frame).zfill(6) + '.jpg'
            self.testset_size += 1

            if info['identity'] not in self.id_images_details.keys():
                self.id_images_details[info['identity']] = []
            self.id_images_details[info['identity']].append(img_name)

            cv2.imwrite(os.path.join(self.save_path, 'test', img_name), cropped_img)


    def select_query_images(self):
        self.query_size = 0
        for id in self.id_images_details.keys():
            if len(self.id_images_details[id]) > 1:
                index = int((len(self.id_images_details[id]) - 1) / 2)
                shutil.move(os.path.join(self.save_path, 'test', self.id_images_details[id][index]),
                            os.path.join(self.save_path, 'query', self.id_images_details[id][index].replace("_c1_", "_c2_")))
                self.query_size += 1

    def run(self):
        self.txt_logger.add_info('for test set:')
        self.txt_logger.add_info('frame interval: {}'.format(self.frame_interval))
        self.txt_logger.add_info('test rate: {}'.format(self.test_rate))

        self.process_gt_result()
        
        self.generate_test_dataset()

        self.select_query_images()
        self.txt_logger.add_info('testset: {} ids, {} images'.format(
            len(self.id_images_details.keys()), (self.testset_size - self.query_size)
        ))
        self.txt_logger.add_info('query: {} images'.format(self.query_size))

        self.txt_logger.output()
        print(len(self.train_id))
        print(len(self.id_images_details.keys()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_sequence_path", type=str, default="./image_sequence/")
    parser.add_argument("--save_path", type=str, default="./reid_dataset/")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--test_rate", type=float, default=0.4, choices=[Range(0.0, 1.0)], help="percentage (frames) of test set")

    return parser.parse_args()


if __name__ == "__main__":
    delete_ids = {
        'MOT16-02': [49, 55, 56, 63, 74, 79],
        'MOT16-11': [67, 68, 69, 70, 71, 72, 73, 75, 86, 87],
        'MOT16-13': [43, 53, 54, 55, 56, 57, 58, 128, 129, 137, 149, 154],
        'MOT16-05': [84, 92, 96, 136, 141, 149]
    }
    rand_seed = 50
    random.seed(rand_seed)

    args = parse_args()

    reidTestDataConverter = ReIDTestDataConverter(args)
    reidTestDataConverter.run()
