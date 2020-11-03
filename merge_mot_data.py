import os
import argparse
import shutil 

from tqdm import tqdm


def get_path_maps(img_filenames, root, source, target, id_map):
    path_map = {}
    source_path = os.path.join(source, root)
    target_path = os.path.join(target, root)
    for name in img_filenames:
        id = int(name.split('.')[0].split('_')[0])
        cam = name.split('.')[0].split('_')[1]
        idx_frame = int(name.split('.')[0].split('_')[2])
        new_id = id_map[id]

        new_name = str(new_id).zfill(5) + '_' + cam + '_' + str(idx_frame).zfill(6) + '.jpg'
        path_map[os.path.join(source_path, name)] = os.path.join(target_path, new_name)

    return path_map


def collect_maps(args):
    id_set = set()
    img_filenames = []
    id_map = {}
    if args.type == "test":
        img_filenames = os.listdir(os.path.join(args.source_path, args.dataset_name, "test"))
    elif args.type == "train":
        img_filenames = os.listdir(os.path.join(args.source_path, args.dataset_name, "train"))

    for name in img_filenames:
        id_set.add(int(name.split('_')[0]))
    id_list = sorted(list(id_set))
    for i, id in enumerate(id_list):
        id_map[id] = args.start_id + i

    path_map = {}
    if args.type == "test":
        img_filenames_query = os.listdir(os.path.join(args.source_path, args.dataset_name, "query"))
        img_filenames_test = os.listdir(os.path.join(args.source_path, args.dataset_name, "test"))
        path_map.update(get_path_maps(img_filenames_query, 'query', os.path.join(args.source_path, args.dataset_name), args.target_path, id_map))
        path_map.update(get_path_maps(img_filenames_test, 'test', os.path.join(args.source_path, args.dataset_name), args.target_path, id_map))
    elif args.type == "train":
        img_filenames_train = os.listdir(os.path.join(args.source_path, args.dataset_name, "train"))
        path_map.update(get_path_maps(img_filenames_train, 'train', os.path.join(args.source_path, args.dataset_name), args.target_path, id_map))

    return path_map


def copy_files(path_map):
    for source_file, target_file in tqdm(path_map.items()):
        shutil.copy(source_file, target_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_id", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--type", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--target_path", type=str, default="./reid_dataset/MOT16/")
    parser.add_argument("--source_path", type=str, default="./reid_dataset/")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    path_map = collect_maps(args)
    copy_files(path_map)
    

