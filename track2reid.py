import os
import cv2
import argparse

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end


class ReIDDataConverter():
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.images_path = os.path.join(args.image_sequence_path, self.dataset_name, "img1")
        assert os.path.isdir(self.images_path), "Images path error"

        self.track_result_path = os.path.join(args.track_result_path, f'{self.dataset_name}.txt')
        assert os.path.isfile(self.track_result_path), "Tracking result path error"

        self.save_path = os.path.join(args.save_path, self.dataset_name)
        self.sampling_rate = args.sampling_rate
        self.partition_rate = args.partition_rate

    def process_track_result(self):
        track_result = []
        id_set = set()
        for line in open(self.track_result_path):
            info = line.split(',')
            idx_frame = info[0]
            identity = info[1]
            # bbox: tlwh
            bbox = (int(info[2]), int(info[3]), int(info[4]), int(info[5]))
            info_dict = {
                'idx_frame': idx_frame,
                'identity': identity,
                'bbox': bbox
            }
            track_result.append(info_dict)
            id_set.add(identity)
        return track_result, id_set

    def run(self):
        self.track_result, self.id_set = self.process_track_result()

        print(len(self.track_result), len(self.id_set))
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_sequence_path", type=str, default="./image_sequence/")
    parser.add_argument("--track_result_path", type=str, default="./output/")
    parser.add_argument("--save_path", type=str, default="./reid_dataset")
    parser.add_argument("--sampling_rate", type=float, default=1, choices=[Range(0.0, 1.0)])
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--partition_rate", type=float, default=0.5, choices=[Range(0.0, 1.0)], help="percentage of training set")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    reidDataConverter = ReIDDataConverter(args)
    reidDataConverter.run()
    