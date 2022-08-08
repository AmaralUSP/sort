import cv2
import math
import argparse
import numpy as np   
import os
from os import listdir
from os.path import isfile, join

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def anns_2_motion_model(dets):
  dets = np.asarray(dets)
  dets[:, 2:4] += dets[:, 0:2] 
  return dets

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)

  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w

  print(f"w {w} h {h} x {x[0]} y {x[1]} s {x[2]} r {x[3]}")
  
  if(score==None):
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def draw_rectangle(img, x1, y1, x2, y2, thickness):
    return cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), thickness)

def euclidianDistance(x0, y0, x1, y1):
    return math.sqrt(math.pow(x0-x1,2)+math.pow(y0-x1,2))
    
def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--setup', dest='setup', help='Setup dataset',action='store_true')

    args = parser.parse_args()
    return args

def set_dataset(path, config):
    path_frames = os.path.join(path, config['Sequence']['imDir'])
    frame_path = [f for f in listdir(path_frames) if isfile(join(path_frames, f))]
    frame_path.sort()

    path_detections = os.path.join(path, 'det', 'det.txt')
    seq_dets = np.loadtxt(path_detections, delimiter=',')

    return seq_dets, path_frames, frame_path