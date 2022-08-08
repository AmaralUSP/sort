# !mkdir -p /data/sets/nuscenes  # Make the directory to store the nuScenes dataset in.

# !wget https://www.nuscenes.org/data/v1.0-mini.tgz  # Download the nuScenes mini split.

# !tar -xf v1.0-mini.tgz -C /data/sets/nuscenes  # Uncompress the nuScenes mini split.

# !pip install nuscenes-devkit &> /dev/null  # Install nuScenes.

import os
import utils
import configparser

from SORT import SORT

IMG_SIZE = (900, 1600)
DATASET_TYPE = 'train'
MOT_VERSION = 'MOT16-02'
total_frames = 0

def main():
    tracker = SORT()

    path = os.path.join('mot_benchmark', DATASET_TYPE, MOT_VERSION)
    
    path_config = os.path.join(path, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(path_config)
    config.sections()

    anns, path_frames, frame_path = utils.set_dataset(path, config)

    # args = parse_args()
    # if args.setup == True:
    #     anns, path_frames, frame_path = set_dataset(path, config)
    
    tracker.track(anns, path_frames, frame_path)

if __name__ == "__main__":
    main()