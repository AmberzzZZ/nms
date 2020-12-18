# soft nms
import numpy as np
import cv2


def matrix_nms(masks, scores, labels, score_thresh=0.1, iou_thresh=0.3, max_detections=200,
               sigma=0.5, method='gaussian'):
    masks = masks.copy()
    scores = scores.copy()
    labels = labels.copy()
    # masks: [N,h,w]
    # scores: [N,1]
    # labels: [N,1]

    # score filter
    valid_indices = np.where(scores>score_thresh)[0]
    masks = masks[valid_indices]
    scores = scores[valid_indices]

    # sort decending
    indices = np.argsort(scores, axis=0)[::-1,0]
    indices = indices[:max_detections]
    masks = masks[indices]
    scores = scores[indices]

    # cal_iou
    iou = cal_miou(masks, masks)     # (N2,N1)

    # triu
    iou = np.triu(iou, k=1)

    # decay factor
    iou_cmax = np.max(iou, axis=0, keepdims=True)
    # if method=='gaussian':
    #     decay = np.exp(-(np.square(iou)-np.square(iou_cmax))/sigma)
    # else:
    #     decay = (1-iou)/(1-iou_cmax)
    # decay = np.min(decay, axis=0)
    if method=='gaussian':
        decay = np.exp(-(np.sum(np.square(iou),axis=0)-np.square(iou_cmax))/sigma)
    else:
        decay = np.prod(1-iou)/(1-iou_cmax)
    decay = decay.reshape(scores.shape)
    scores = scores * decay

    # thresh filter
    picked = np.where(scores>score_thresh)

    return masks[picked[0]], scores[picked], labels[picked]


def cal_miou(mask1, mask2, epsilon=1e-5):
    mask1 = mask1.copy()
    mask2 = mask2.copy()
    # mask1: [N1,h,w]
    # mask2: [N2,h,w]

    mask1 = mask1.reshape((mask1.shape[0], -1))
    mask2 = mask2.reshape((mask2.shape[0], -1))

    inter_area = mask1.dot(mask2.T)
    mask_area1 = np.sum(mask1, axis=1).reshape(mask1.shape[0],-1)
    mask_area2 = np.sum(mask2, axis=1).reshape(-1, mask2.shape[0])

    iou = inter_area / (mask_area1 + mask_area2 - inter_area + epsilon)

    return iou


def cal_mmi(mask1, mask2, epsilon=1e-5):
    # mask1: [N1,h,w]
    # mask2: [N2,h,w]

    mask1 = mask1.reshape((mask1.shape[0], -1))
    mask2 = mask2.reshape((mask2.shape[0], -1))

    inter_area = mask1.dot(mask2.T)
    mask_area1 = np.sum(mask1, axis=1).reshape(mask1.shape[0],-1)
    mask_area2 = np.sum(mask2, axis=1).reshape(-1, mask2.shape[0])

    iou = np.maximum(inter_area/mask_area1, inter_area/mask_area2)

    return iou


if __name__ == '__main__':

    mask1 = np.zeros((100,100))
    mask1[20:40, 30:50] = 1

    mask2 = np.zeros((100,100))
    mask2[25:40, 38:55] = 1

    mask3 = np.zeros((100,100))
    mask3[45:88, 45:50] = 1

    mask4 = np.zeros((100,100))
    mask4[15:45, 28:55] = 1

    masks = np.stack([mask1, mask2, mask3, mask4], axis=0)
    scores = np.array([0.9, 0.6, 0.8, 0.95]).reshape((-1,1))

    mask, score, _ = matrix_nms(masks, scores, scores, iou_thresh=0.3, score_thresh=0.5)

    print("result: ")
    print(score)

    # vis
    canvas = np.ones((100,100))
    cv2.rectangle(canvas, (30,20), (50,40), 0, 2)
    # cv2.rectangle(canvas, (30,20), (50,40), 0.5, -1)
    cv2.rectangle(canvas, (38,25), (55,40), 0, 2)
    # cv2.rectangle(canvas, (38,25), (55,40), 0.5, -1)
    cv2.rectangle(canvas, (45,45), (50,88), 0, 2)
    # cv2.rectangle(canvas, (45,45), (50,88), 0.5, -1)
    cv2.rectangle(canvas, (28,15), (55,45), 0, 2)
    # cv2.rectangle(canvas, (28,15), (55,45), 0.5, -1)
    cv2.imshow("before nms", canvas)
    cv2.waitKey(0)

    canvas = np.ones((100, 100))
    for m in mask:
        coords = np.where(m>0)
        xmin, xmax, ymin, ymax = np.min(coords[1]), np.max(coords[1]), np.min(coords[0]), np.max(coords[0])
        cv2.rectangle(canvas, (xmin,ymin), (xmax,ymax), 0, 2)
        cv2.rectangle(canvas, (xmin,ymin), (xmax,ymax), 0.5, -1)
    cv2.imshow("after nms", canvas)
    cv2.waitKey(0)








