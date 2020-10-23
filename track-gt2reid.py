"""
@author: tenghehan

处理 tracking 的 ground truth，生成对应的 reid 数据集.
"""
import os
import shutil

import cv2
import argparse
import random

from tqdm import tqdm
from utils.log import get_logger
from utils.txt_logger import txt_logger


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end


class ReIDDataConverter():
    def __init__(self, args):
        self.logger = get_logger('root')
        self.txt_logger = txt_logger(os.path.join(args.save_path, args.dataset_name, 'info.txt'))

        self.dataset_name = args.dataset_name
        self.images_path = os.path.join(args.image_sequence_path, self.dataset_name, "img1")
        assert os.path.isdir(self.images_path), "Images path error"

        self.gt_file_path = os.path.join(args.image_sequence_path, self.dataset_name, "gt/gt.txt")
        assert os.path.isfile(self.gt_file_path), "gt file path error"

        self.save_path = os.path.join(args.save_path, self.dataset_name)
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(os.path.join(self.save_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'query'), exist_ok=True)

        self.sampling_rate = args.sampling_rate
        self.sampled_imgs_filenames = self.sample_frames()

        self.partition_rate = args.partition_rate
        self.gt_result = []
        self.id_set = {'train_id_set': set(), 'test_id_set': set()}
        self.id_list = []
        self.id_images_details = {}

        first_img = cv2.imread(os.path.join(self.images_path, self.sampled_imgs_filenames[0]))
        self.im_width = first_img.shape[1]
        self.im_height = first_img.shape[0]

    def partition_train_test(self, id_list):
        total_num = len(id_list)
        train_list = random.sample(id_list, int(total_num * self.partition_rate))
        train_set = set(train_list)
        self.id_set = {
            'train_id_set': train_set,
            'test_id_set': set(id_list) - train_set,
        }

        for identity in self.id_list:
            self.id_images_details[identity] = []

    def sample_frames(self):
        imgs_filenames = sorted(os.listdir(self.images_path))
        sampled_imgs_filenames = []

        for img_filename in imgs_filenames:
            if random.random() <= self.sampling_rate:
                sampled_imgs_filenames.append(img_filename)

        return sampled_imgs_filenames

    def process_gt_result(self):
        for line in open(self.gt_file_path):
            info = line.split(',')
            idx_frame = int(info[0])
            identity = int(info[1])
            # bbox: tlwh
            bbox = (int(info[2]), int(info[3]), int(info[4]), int(info[5]))
            consider = int(info[6])
            type = int(info[7])
            visiblity = float(info[8])
            info_dict = {
                'idx_frame': idx_frame,
                'identity': identity,
                'bbox': bbox
            }

            if consider == 0 or type != 1 or visiblity < 0.6:
                continue

            self.gt_result.append(info_dict)
            if identity not in self.id_list:
                self.id_list.append(identity)

        self.partition_train_test(self.id_list)
        return self.gt_result, self.id_set

    def cal_train_test_ids(self):
        train_ids, test_ids = 0, 0
        for id in self.id_set['train_id_set']:
            if len(self.id_images_details[id]) > 0:
                train_ids += 1
        for id in self.id_set['test_id_set']:
            if len(self.id_images_details[id]) > 0:
                test_ids += 1
        return train_ids, test_ids

    def generate_reid_dataset(self):
        trainset_size = 0
        testset_size = 0
        for gt in tqdm(self.gt_result):
            # if the frame is sampled
            idx_frame = gt['idx_frame']

            if f'{str(idx_frame).zfill(6)}.jpg' not in self.sampled_imgs_filenames:
                continue

            # read frame image
            frame_path = os.path.join(self.images_path, f'{str(idx_frame).zfill(6)}.jpg')
            frame = cv2.imread(frame_path)

            # crop the person area from the whole image
            x1 = max(gt['bbox'][0], 0)
            y1 = max(gt['bbox'][1], 0)
            x2 = min(gt['bbox'][0] + gt['bbox'][2], self.im_width)
            y2 = min(gt['bbox'][1] + gt['bbox'][3], self.im_height)
            if x2 <= x1 or y2 <= y1:
                continue
            cropped_img = frame[y1:y2, x1:x2]
            img_name = str(gt['identity']).zfill(5) + '_c1_' + str(idx_frame).zfill(6) + '.jpg'
            self.id_images_details[gt['identity']].append(img_name)

            # save the person image into reid train/test dataset
            if gt['identity'] in self.id_set['train_id_set']:
                cv2.imwrite(os.path.join(self.save_path, 'train', img_name), cropped_img)
                trainset_size += 1
            else:
                cv2.imwrite(os.path.join(self.save_path, 'test', img_name), cropped_img)
                testset_size += 1

        train_ids, test_ids = self.cal_train_test_ids()
        return train_ids, test_ids, trainset_size, testset_size

    def select_query_images(self):
        query_size = 0
        for identity in self.id_set['test_id_set']:
            if len(self.id_images_details[identity]) > 1:
                shutil.move(os.path.join(self.save_path, 'test', self.id_images_details[identity][0]),
                            os.path.join(self.save_path, 'query', self.id_images_details[identity][0].replace("_c1_", "_c2_")))
                query_size += 1

        return query_size

    def run(self):
        self.txt_logger.add_info('sampling rate: {}'.format(self.sampling_rate))
        self.txt_logger.add_info('partition rate: {}'.format(self.partition_rate))

        self.gt_result, self.id_set = self.process_gt_result()

        self.logger.info('generating reid dataset ...')
        train_ids, test_ids, trainset_size, testset_size = self.generate_reid_dataset()

        self.logger.info('selecting query images from test dataset ...')
        query_size = self.select_query_images()

        self.logger.info('reid dataset {} generated'.format(self.dataset_name))

        self.txt_logger.add_info('trainset: {} identities, {} images'.format(
            train_ids, trainset_size))
        self.txt_logger.add_info('testset: {} identities, {} images'.format(
            test_ids, (testset_size -  query_size)))
        self.txt_logger.add_info('query: {} images'.format(query_size))

        self.txt_logger.output()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_sequence_path", type=str, default="./image_sequence/")
    parser.add_argument("--save_path", type=str, default="./reid_dataset_gt/")
    parser.add_argument("--sampling_rate", type=float, default=1, choices=[Range(0.0, 1.0)])
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--partition_rate", type=float, default=0.8, choices=[Range(0.0, 1.0)], help="percentage (identity) of training set")

    return parser.parse_args()


if __name__ == "__main__":
    rand_seed = 50
    random.seed(rand_seed)

    args = parse_args()
    reidDataConverter = ReIDDataConverter(args)
    reidDataConverter.run()
    