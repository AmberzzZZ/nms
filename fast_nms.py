# fast nms
import numpy as np
import cv2


def fast_nms(boxes, scores, score_thresh=0.1, iou_thresh=0.3, max_detections=200):
    # boxes: [N,4], x1y1x2y2
    # scores: [N,1]

    # score filter
    valid_indices = np.where(scores>score_thresh)[0]
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]

    # sort decending
    indices = np.argsort(scores, axis=0)[::-1,0]
    indices = indices[:max_detections]
    boxes = boxes[indices]
    scores = scores[indices]

    # cal_iou
    iou = cal_iou(boxes[...,:4], boxes[...,:4])     # (N2,N1)

    # triu
    iou = np.triu(iou, k=1)

    # thresh filter
    picked = np.max(iou, axis=0)    # (N1,)
    picked = np.where(picked<iou_thresh)

    return boxes[picked], scores[picked]


def cal_iou(boxes1, boxes2, epsilon=1e-5):
    # boxes1: [N1,4], x1y1x2y2
    # boxes2: [N2,4], x1y1x2y2

    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes2 = np.expand_dims(boxes2, axis=0)

    inter_mines = np.maximum(boxes1[...,:2], boxes2[...,:2])    # [N1,N2,2]
    inter_maxes = np.minimum(boxes1[...,2:], boxes2[...,2:])
    inter_wh = np.maximum(inter_maxes - inter_mines, 0.)
    inter_area = inter_wh[...,0] * inter_wh[...,1]

    box_area1 = (boxes1[...,2]-boxes1[...,0]) * (boxes1[...,3]-boxes1[...,1])
    box_area1 = np.tile(box_area1, [1,np.shape(boxes2)[0]])
    box_area2 = (boxes2[...,2]-boxes2[...,0]) * (boxes2[...,3]-boxes2[...,1])
    box_area2 = np.tile(box_area2, [np.shape(boxes1)[0],1])

    iou = inter_area / (box_area1 + box_area2 - inter_area + epsilon)

    return iou


if __name__ == '__main__':

    bboxes = np.array([[30,12,50,28,0.9],
                       [25,10,52,25,0.95],
                       [27,11,55,23,0.8],
                       [33,9,44,26,0.6],
                       [80,88,100,122,0.9]])
    box, score = fast_nms(bboxes[...,:4], bboxes[...,-1:], iou_thresh=0.3, score_thresh=0.5)

    print("result: ")
    print(box)
    print(score)

    # # vis
    # canvas = np.zeros((150, 150))
    # cv2.rectangle(canvas, (30,12), (50,28), 255, 1)
    # cv2.rectangle(canvas, (25,10), (52,25), 255, 1)
    # cv2.rectangle(canvas, (27,11), (55,23), 255, 1)
    # cv2.rectangle(canvas, (33,9), (44,26), 255, 1)
    # cv2.rectangle(canvas, (80,88), (100,122), 255, 1)
    # cv2.imshow("before nms", canvas)
    # cv2.waitKey(0)

    # canvas = np.zeros((150, 150))
    # for b in box:
    #     cv2.rectangle(canvas, (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), 255, 1)
    # cv2.imshow("after nms", canvas)
    # cv2.waitKey(0)
