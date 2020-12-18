from hard_nms import hard_nms
from soft_nms import soft_nms
from diou_nms import diou_nms
from fast_nms import fast_nms
from cluster_nms import cluster_nms
from matrix_nms import matrix_nms
import numpy as np
import cv2

# np.random.seed(0)


if __name__ == '__main__':

    input_shape = (480,640)     # hw
    stride = 80
    anchors = np.array([[20,20],[40,15],[15,40]]).reshape((1,1,3,2))

    # fake outputs
    grid_h, grid_w = input_shape[0]//stride, input_shape[1]//stride
    output_xy = np.random.uniform(-0.5, 0.5, (grid_h,grid_w,3,2))
    output_wh = np.random.uniform(-10000, 10000, (grid_h,grid_w,3,2))
    output_cls = np.random.uniform(0,0.6,(grid_h,grid_w,3,10))
    output = np.concatenate([output_xy, output_wh, output_cls], axis=-1)
    print("faking output: ", output.shape)

    # labels & probs
    cls = np.argmax(output[:,:,:,4:], axis=-1)
    print("pred cls: ", cls.shape)
    print(np.unique(cls), np.max(cls), np.min(cls))
    prob = np.max(output[:,:,:,4:], axis=-1)
    print("pred probs", prob.shape, np.max(prob), np.min(prob))

    # grid offset
    grid_offset_x, grid_offset_y = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid_coords = np.stack([grid_offset_x, grid_offset_y], axis=-1)
    grid_coords = np.expand_dims(grid_coords, axis=2)
    print("grid_coords: ", grid_coords.shape)
    center_coords = grid_coords + 0.5

    # pred_xcycwh_abs
    input_shape = np.array(input_shape)[::-1]
    pred_xcyc_abs = output[:,:,:,:2] * anchors + (center_coords / [grid_w, grid_h] * input_shape)
    pred_wh_abs = np.exp(output[:,:,:,:2]) * anchors
    print("grid preds abs: ", pred_xcyc_abs.shape, pred_wh_abs.shape)

    pred_x1y1 = pred_xcyc_abs - pred_wh_abs/2
    pred_x2y2 = pred_xcyc_abs + pred_wh_abs/2
    boxes = np.concatenate([pred_x1y1, pred_x2y2], axis=-1).reshape((-1,4))
    print(boxes.shape)
    scores = np.reshape(prob, (-1,1))
    print(scores.shape)
    labels = np.reshape(cls, (-1,1))
    print(labels.shape)

    # # filter the outing points
    # indices = np.where(boxes[:,0]<0)
    # indices = np.where(boxes[:,2]>500)
    # labels[indices] = 0

    # vis
    input_shape = np.array(input_shape)[::-1]
    canvas = np.zeros(input_shape)
    center_points = (center_coords.reshape(grid_h,grid_w,2) * stride).astype(np.int)
    print(center_points.shape)
    for i in range(grid_h):
        for j in range(grid_w):
            cv2.circle(canvas, (center_points[i,j,0], center_points[i,j,1]), 1, 1, 1)
            # for k in range(3):
            #     w,h = anchors[0,0,k]
            #     x1,x2 = int(center_points[i,j,0]-w/2.), int(center_points[i,j,0]+w/2.)
            #     y1,y2 = int(center_points[i,j,1]-h/2.), int(center_points[i,j,1]+h/2.)
            #     cv2.rectangle(canvas, (x1,y1), (x2,y2), 1, 1)
            for k in range(3):
                w,h = pred_wh_abs[i,j,k]
                xc,yc = pred_xcyc_abs[i,j,k]
                cv2.rectangle(canvas, (int(xc-w/2),int(yc-h/2)), (int(xc+w/2),int(yc+h/2)), 1, 1)
    cv2.imshow("before nms", canvas)
    cv2.waitKey(0)

    for nms_method in [hard_nms, soft_nms, diou_nms, fast_nms, cluster_nms]:
        boxes_, scores_, labels_ = nms_method(boxes, scores, labels,
                                              score_thresh=0.3, iou_thresh=0.2, max_detections=100)
        print("method %s outing boxes" % nms_method, boxes_.shape, scores_.shape, labels_.shape)
        boxes_ = boxes_.astype(np.int)

        # vis
        canvas = np.zeros(input_shape)
        for i in range(boxes_.shape[0]):
            x1,y1,x2,y2 = boxes_[i]
            cv2.rectangle(canvas, (x1,y1), (x2,y2), 1, 1)
        cv2.imshow(str(nms_method), canvas)
        cv2.waitKey(0)






