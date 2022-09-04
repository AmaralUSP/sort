import os
import utils
import configparser
import shutil
import json

from SORT import SORT
from tqdm import tqdm 

IMG_SIZE = (900, 1600)
DATASET_TYPE = 'train'
MOT_VERSION = 'MOT16-02'
total_frames = 0

def main():
    versions = os.listdir(os.path.join('mot_benchmark', DATASET_TYPE))
    results = {}
    scene_path = "/home/wellington/Videos/EKF"

    for version in versions:
        tracker = SORT()

        path = os.path.join('mot_benchmark', DATASET_TYPE, version)
        curr_scene_path = os.path.join(scene_path, version)
        
        if os.path.exists(curr_scene_path):
            shutil.rmtree(curr_scene_path)

        os.mkdir(curr_scene_path)
        
        path_config = os.path.join(path, 'seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(path_config)
        config.sections()

        anns, path_frames, frame_path = utils.set_dataset(path, config)
        anns = anns[anns[:, 0].argsort()]
        
        results[version] = tracker.track(anns, path_frames, frame_path, curr_scene_path)

    # with open("sort_results_EKF.json", 'w') as f:
    #     json.dump(results, f)

if __name__ == "__main__":
    main()