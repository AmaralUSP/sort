import cv2
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from soupsieve import select

from KalmanObject import KalmanObject
import utils

MIN_ASSOCIATIONS = 2
IOU_THRESHOLD = 0.7
THICKNESS = 2

class SORT:    
    __minAssociations = MIN_ASSOCIATIONS
    __IoUTrashold = IOU_THRESHOLD
    __countObjs = 0

    def __init__(self):
        self.timeWithoutDetections = 0    
        self.objTracker = []
        self.scenariosObj = []

    def __associateObjects(self, objs, dets):
        iou_matrix = utils.iou_batch(objs, dets)
        matched_index = utils.linear_assignment(-iou_matrix)
        unmatch_trks = []
        unmatch_dets = []

        for index, c_obj in enumerate(self.objTracker):
            if index not in matched_index[:, 0]:
                self.objTracker.pop(index) 
                unmatch_trks.append(index) # guardar em um historico global de deteccao

        for index, c_det in enumerate(dets):
            if index not in matched_index[:, 1] and iou_matrix[:, index].max() > IOU_THRESHOLD:
                init_pos = utils.convert_bbox_to_z(c_det)
                self.objTracker.append(KalmanObject(2, 500, init_pos)) 
                unmatch_dets.append(index)

        return matched_index, unmatch_dets, unmatch_trks

    def __update(self, matched_index, unmatch_dets, dets):
        for index, c_obj in enumerate(self.objTracker):
            if index in matched_index[:, 0]:
                test = np.expand_dims(dets[matched_index[index, 1]], axis=1)
                c_obj.update(test)


    def __show_frame(self, img):
        # Faz as anotacoes no frame (Utils method)
        for bboxx in self.objTracker:
            corners = utils.convert_x_to_bbox(bboxx.getState())
            print(corners)
            # print(corners.shape)
            img = utils.draw_rectangle(img, int(corners[0][0]), int(corners[0][1]), int(corners[0][2]), int(corners[0][3]), THICKNESS)

        return img   

    def track(self, anns, path_frames, frame_path):
        '''
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        '''

        current_obj = 0
        for frame_num, frame in enumerate(frame_path[:-1], 1):    
            # ajusta a lista de detecoes para o formato para o Kalman (Utils method)
            dets = [] # deteccoes
            trks = [] # objetos rastreados
            unmatch_trks = [] # apos a detccao os objetos rastreados que n deram match
            match_trks = [] # apos a detccao os objetos rastreados que deram match
            int_pos = []

            img = cv2.imread(os.path.join(path_frames, frame))

            while anns[current_obj][0] == frame_num:
                dets.append(np.array(anns[current_obj][2:6]))
                current_obj = current_obj + 1

            dets = utils.anns_2_motion_model(dets)

            if len(self.objTracker) == 0:
                for curr_track in dets:
                    init_pos = utils.convert_bbox_to_z(curr_track)
                    self.objTracker.append(KalmanObject(2, 500, init_pos))
                    # dar ids para os objetos

                # show images then continue
                img = self.__show_frame(img)

                # Mostra o frame
                cv2.imshow("cena", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                continue

            # fazer a predicao
            for x in self.objTracker:
                x.predict()

            objs = []
            for x in self.objTracker:
                objs.append(utils.convert_x_to_bbox(x.getState()))

            objs = np.concatenate(objs, axis=0)

            matched_index, unmatch_dets, _  = self.__associateObjects(objs, dets)
            # unmatch_ids = self.__associateObjects(objs, dets)

            # atualiza a lista de objetos
            # Faz as predicoes utilizando Kalman (KalmanObject Method)
            # for x in self.objTracker:
            #     print("antes update")
            #     print(x.getState())

            self.__update(matched_index, unmatch_dets, dets)

            # for x in self.objTracker:
            #     print("dps update")
            #     print(x.getState())

            img = self.__show_frame(img)
            # Mostra o frame
            cv2.imshow("cena", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
