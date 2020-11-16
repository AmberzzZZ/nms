blog: https://amberzzzz.github.io/2020/10/29/nms/


0. hard nms
    score filter + sort + tranverse

    优化空间在iou的计算上：
        可以通过矩阵操作并行计算N个box之间的iou，
        tranverse的逻辑一样，维护一个running indices，
        始终提取第一个index的box作为current，提取对应行其余indices对应的iou

    缺点：
        将相邻检测框的分数均强制归零，如果真实场景是重叠场景，就会严重漏检


1. soft nms
    调整检测框分数重置逻辑，不再粗暴地直接将高重叠预测框置零
    gaussian / linear



2. DIoU nms
    调整iou计算方式，diou兼顾了重叠度和框的距离


3. fast nms
    yoloact提出，用矩阵操作替代遍历操作，所有框是同时被filter掉的，而非依次遍历删除
    与hard nms的结果是不一样的，有精度损失，会比hard nms抑制更多的框
    针对mask nms提出，mask AP的下降比较轻微，对目标检测的box AP会下降更多


4. cluster nms
    ciou团队提出，弥补fast nms，是少量迭代次数的fast nms
    结果与hard nms一样，效率稍微低于fast nms


5. matrix nms
    solov2提出，用矩阵操作替代遍历操作，
    decayfactor基于原论文的思路实现，和原论文的伪代码和源代码不一样，原论文的实现有问题，decay永远大于等于1


## params
    nms_thresh: 用来删除框的iou thresh，值越小删除的框越多
    score_thresh: 如果有confidence，score应该是conf*cls_prob
    max_detections: 保留最大框数目
    class_specific: 对每个类分别做nms




