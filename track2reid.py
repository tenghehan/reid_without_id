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

        self.track_result_path = os.path.join(args.track_result_path, f'{self.dataset_name}.txt')
        assert os.path.isfile(self.track_result_path), "Tracking result path error"

        self.save_path = os.path.join(args.save_path, self.dataset_name)
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(os.path.join(self.save_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'query'), exist_ok=True)

        self.sampling_rate = args.sampling_rate
        self.sampled_imgs_filenames = self.sample_frames()

        self.partition_rate = args.partition_rate
        self.track_result = []
        self.id_set = {'train_id_set': set(), 'test_id_set': set()}
        self.id_images_details = {}

    def partition_train_test(self, identity):
        if identity in self.id_set['train_id_set'] or identity in self.id_set['test_id_set']:
            return

        if random.random() <= self.partition_rate:
            self.id_set['train_id_set'].add(identity)
        else:
            self.id_set['test_id_set'].add(identity)

        if identity not in self.id_images_details.keys():
            self.id_images_details[identity] = []

    def sample_frames(self):
        imgs_filenames = sorted(os.listdir(self.images_path))
        sampled_imgs_filenames = []

        for img_filename in imgs_filenames:
            if random.random() <= self.sampling_rate:
                sampled_imgs_filenames.append(img_filename)

        return sampled_imgs_filenames

    def process_track_result(self):
        for line in open(self.track_result_path):
            info = line.split(',')
            idx_frame = int(info[0]) + 1
            identity = int(info[1])
            # bbox: tlwh
            bbox = (int(info[2]), int(info[3]), int(info[4]), int(info[5]))
            info_dict = {
                'idx_frame': idx_frame,
                'identity': identity,
                'bbox': bbox
            }
            self.track_result.append(info_dict)
            self.partition_train_test(identity)
        return self.track_result, self.id_set

    def generate_reid_dataset(self):
        trainset_size = 0
        testset_size = 0
        for track in tqdm(self.track_result):
            # if the frame is sampled
            idx_frame = track['idx_frame']

            if f'{str(idx_frame).zfill(6)}.jpg' not in self.sampled_imgs_filenames:
                continue

            # read frame image
            frame_path = os.path.join(self.images_path, f'{str(idx_frame).zfill(6)}.jpg')
            frame = cv2.imread(frame_path)

            # crop the person area from the whole image
            x1 = track['bbox'][0]
            y1 = track['bbox'][1]
            x2 = track['bbox'][0] + track['bbox'][2]
            y2 = track['bbox'][1] + track['bbox'][3]
            cropped_img = frame[y1:y2, x1:x2]
            img_name = str(track['identity']).zfill(5) + '_' + str(idx_frame).zfill(6) + '.jpg'
            self.id_images_details[track['identity']].append(img_name)

            # save the person image into reid train/test dataset
            if track['identity'] in self.id_set['train_id_set']:
                cv2.imwrite(os.path.join(self.save_path, 'train', img_name), cropped_img)
                trainset_size += 1
            else:
                cv2.imwrite(os.path.join(self.save_path, 'test', img_name), cropped_img)
                testset_size += 1

        return trainset_size, testset_size

    def select_query_images(self):
        query_size = 0
        for identity in self.id_set['test_id_set']:
            if len(self.id_images_details[identity]) > 1:
                shutil.move(os.path.join(self.save_path, 'test', self.id_images_details[identity][0]),
                            os.path.join(self.save_path, 'query', self.id_images_details[identity][0]))
                query_size += 1

        return query_size

    def run(self):
        self.txt_logger.add_info('sampling rate: {}'.format(self.sampling_rate))
        self.txt_logger.add_info('partition rate: {}'.format(self.partition_rate))

        self.track_result, self.id_set = self.process_track_result()

        self.logger.info('generating reid dataset ...')
        trainset_size, testset_size = self.generate_reid_dataset()

        self.logger.info('selecting query images from test dataset ...')
        query_size = self.select_query_images()

        self.logger.info('reid dataset {} generated'.format(self.dataset_name))

        self.txt_logger.add_info('trainset: {} identities, {} images'.format(
            len(self.id_set['train_id_set']), trainset_size))
        self.txt_logger.add_info('testset: {} identities, {} images'.format(
            len(self.id_set['test_id_set']), (testset_size -  query_size)))
        self.txt_logger.add_info('query: {} images'.format(query_size))

        self.txt_logger.output()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_sequence_path", type=str, default="./image_sequence/")
    parser.add_argument("--track_result_path", type=str, default="./output/")
    parser.add_argument("--save_path", type=str, default="./reid_dataset/")
    parser.add_argument("--sampling_rate", type=float, default=1, choices=[Range(0.0, 1.0)])
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--partition_rate", type=float, default=0.5, choices=[Range(0.0, 1.0)], help="percentage (identity) of training set")

    return parser.parse_args()


if __name__ == "__main__":
    rand_seed = 50
    random.seed(rand_seed)

    args = parse_args()
    reidDataConverter = ReIDDataConverter(args)
    reidDataConverter.run()
    