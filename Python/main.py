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
    anns = anns[anns[:, 0].argsort()]

    # args = parse_args()
    # if args.setup == True:
    #     anns, path_frames, frame_path = set_dataset(path, config)
    
    tracker.track(anns, path_frames, frame_path)

if __name__ == "__main__":
    main()