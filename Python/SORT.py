from nis import match
import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from soupsieve import select
from tqdm import tqdm
from KalmanObject import KalmanObject
import utils

MIN_ASSOCIATIONS = 2
IOU_THRESHOLD = 0.7
THICKNESS = 2
TIME_WITHOUT_DETS = 5

class SORT:    
    __minAssociations = MIN_ASSOCIATIONS
    __IoUTrashold = IOU_THRESHOLD
    __countObjs = 0

    def __init__(self):
        self.timeWithoutDetections = 0    
        self.objTracker = []
        self.scenariosObj = []
        self.count = 0
        
        self.fp = 0
        self.tp = 0
        self.acuracy = 0
        self.MOTA = 0

    def __associateObjects(self, objs, dets):
        iou_matrix = utils.iou_batch(dets, objs)
        matched_index = utils.linear_assignment(-iou_matrix)
        unmatch_trks = [] # o que tinha e nao associei
        unmatch_dets = [] # o que eu detectei e nao associei
        to_del = []

        for index, c_obj in enumerate(self.objTracker):
            if index not in matched_index[:, 1]:
                unmatch_trks.append(c_obj.unique_id) # guardar em um historico global de deteccao

        for index, c_det in enumerate(dets):
            if (index not in matched_index[:, 0]):
                init_pos = utils.convert_bbox_to_z(c_det)
                self.objTracker.append(KalmanObject(2, 500, init_pos, self.count))
                self.count = self.count + 1
                unmatch_dets.append(index)

        match = []
        for index, m in enumerate(matched_index):
            if iou_matrix[m[0], m[1]] < IOU_THRESHOLD:
                unmatch_dets.append(m[0])
                unmatch_trks.append(m[1]) 

            else:
                match.append([m[0], self.objTracker[m[1]].unique_id])

        match = np.asarray(match)
        unmatch_dets = np.asarray(unmatch_dets)
        unmatch_trks = np.asarray(unmatch_trks)

        return match, unmatch_dets, unmatch_trks

    def __update(self, matched_index, unmatch_dets, dets):
        for c_obj in self.objTracker:
            if c_obj.unique_id in matched_index[:, 1]:
                id = np.where(matched_index[:, 1] == c_obj.unique_id)[0][0]
                test = np.expand_dims(dets[matched_index[id, 0]], axis=1)
                test = utils.convert_bbox_to_z(test)
                c_obj.update(test)

    def __show_frame(self, img):
        # Faz as anotacoes no frame (Utils method)
        for bboxx in self.objTracker:
            corners = utils.convert_x_to_bbox(bboxx.getState())
            img = utils.draw_rectangle(img, 
                                       int(corners[0][0]), 
                                       int(corners[0][1]),
                                       int(corners[0][2]), 
                                       int(corners[0][3]), 
                                       THICKNESS, 
                                       bboxx.unique_id, 
                                       (int(corners[0][2]), int(corners[0][3])), color=bboxx.color)

        return img 

    def __show_frame_gt(self, dets, ids, img):
        # Faz as anotacoes no frame (Utils method)
        for id, bboxx in enumerate(dets):
            img = utils.draw_rectangle(img,
                                       int(bboxx[0]), 
                                       int(bboxx[1]), 
                                       int(bboxx[2]), 
                                       int(bboxx[3]), 
                                       THICKNESS, int(ids[id]), 
                                       (int(bboxx[0]), int(bboxx[1])))

        return img  

    def calculate_area(self, det):
        return det[2] * det[3]

    def calculate_idsw(self, matched_index, ids, id_gt, metrics):
        # idsw
        for d in matched_index: # d [dets, unique_id]
            if ids[d[0]] in id_gt:
                if d[1] != id_gt[ids[d[0]]][-1]:
                    id_gt[ids[d[0]]].append(d[1])
                    metrics["IDSW"] += 1
            else:
                id_gt[ids[d[0]]] = [d[1]]
                
        print(matched_index.shape, np.shape(ids))
    def calculate_fp_fn(self, metrics, unmatch_dets, matched_index, dets):
        # fp
        for obj in self.objTracker:
            if obj.frames_without_dets > 0:
                metrics["FP"] += 1

        # fn
        for d, i in enumerate(dets):
            if (d not in matched_index[:, 1]) and (d not in unmatch_dets):
                metrics["FN"] += 1

        metrics["GT"] += len(dets)
            

    def valid_obs(self, det):
        return self.calculate_area(det) > 2500

    def track(self, anns, path_frames, frame_path):
        '''
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        '''

        current_obj = 0
        metrics = {"FP" : 0, "FN" : 0, "IDSW" : 0, "MOTA" : 0, "GT" : 0}
        id_gt = {}

        for frame_num, frame in tqdm(enumerate(frame_path[:-1], 1), total=len(frame_path[:-1])):    
            # ajusta a lista de detecoes para o formato pasimra o Kalman (Utils method)
            dets = [] # deteccoes
            ids = []

            img = cv2.imread(os.path.join(path_frames, frame))

            while anns[current_obj][0] == frame_num:
                without_noise = np.array(anns[current_obj][2:6])

                if self.valid_obs(without_noise):
                    with_noise = without_noise + np.random.normal(0, 1, without_noise.shape)
                    dets.append(with_noise)
                    ids.append(anns[current_obj][1])

                current_obj = current_obj + 1

            dets = utils.anns_2_motion_model(dets)

            if len(self.objTracker) == 0:
                for curr_track in dets:
                    init_pos = utils.convert_bbox_to_z(curr_track)
                    if init_pos[2] > 2500:
                        self.objTracker.append(KalmanObject(2, 500, init_pos, self.count))
                        self.count = self.count + 1

                img = self.__show_frame(img)

                cv2.imwrite(f"cena/{frame_num}.png", img)
                continue

            to_del = []
            for index, x in enumerate(self.objTracker):
                if not x.is_valid(TIME_WITHOUT_DETS):
                    to_del.append(index)
                    continue
                
                x.predict()
                
                if np.any(np.isnan(utils.convert_x_to_bbox(x.getState()))):
                    to_del.append(index)

            for x in reversed(to_del):
                self.objTracker.pop(x)            

            objs = []
            for x in self.objTracker:
                objs.append(utils.convert_x_to_bbox(x.getState()))

            objs = np.concatenate(objs, axis=0)

            matched_index, unmatch_dets, unmatch_trks = self.__associateObjects(objs, dets)
            
            self.calculate_idsw(matched_index, ids, id_gt, metrics)
            self.__update(matched_index, unmatch_dets, dets)
            # self.calculate_fp_fn(metrics, unmatch_dets, matched_index, dets)

            print(metrics)

            img = self.__show_frame_gt(dets, ids, img)
            img = self.__show_frame(img)

            cv2.imwrite(f"cena/{frame_num}.png", img)

