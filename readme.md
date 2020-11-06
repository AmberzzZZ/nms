0. hard nms
    score filter + sort + tranverse
    优化空间在iou的计算上：
        可以通过矩阵操作并行计算N个box之间的iou，
        tranverse的逻辑一样，维护一个running indices，
        始终提取第一个index的box作为current，提取对应行其余indices对应的iou


1. NMS


2. Fast NMS


3. mask NMS



## params
    nms_thresh: 用来删除框的iou thresh，值越小删除的框越多
    score_thresh: 如果有confidence，score应该是conf*cls_prob
    max_detections: 保留最大框数目
    class_specific: 对每个类分别做nms




