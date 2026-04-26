# 读取Yaml文件
def read_yaml(path):
    import yaml
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return f'错误：{path}不存在!'
    except yaml.YAMLError as e:
        return f'{path}文件解析错误！{e}'

# 读取JSON文件
def read_json(path):
    import json
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return f'错误：{path}不存在!'
    except json.JSONDecodeError as e:
        return f'{path}文件解析错误！{e}'

# 保存模型权重
def save_weight(model, path):
    import paddle
    paddle.save(model.state_dict(), f'{path}.pdparams')

# 加载模型权重
def load_weight(model, path):
    import paddle
    state_dict = paddle.load(path)
    model.set_dict(state_dict)

# 计算交并比
def calc_iou(pred_box, gt_box):
    import paddle
    # gt_box: [batch, s, s, 4（x, y, w, h）]
    gt_x, gt_y = gt_box[..., 0], gt_box[..., 1]
    gt_w, gt_h = gt_box[..., 2], gt_box[..., 3]
    gt_x1, gt_y1 = gt_x - gt_w / 2, gt_y - gt_h / 2
    gt_x2, gt_y2 = gt_x + gt_w / 2, gt_y + gt_h / 2

    pred_x, pred_y = pred_box[..., 0], pred_box[..., 1]
    pred_w, pred_h = pred_box[..., 2], pred_box[..., 3]
    pred_x1, pred_y1 = pred_x - pred_w / 2, pred_y - pred_h / 2
    pred_x2, pred_y2 = pred_x + pred_w / 2, pred_y + pred_h / 2

    # 计算交集的左上角和右下角的x,y坐标
    inter_x1, inter_y1 = paddle.maximum(gt_x1, pred_x1), paddle.maximum(gt_y1, pred_y1)
    inter_x2, inter_y2 = paddle.minimum(gt_x2, pred_x2), paddle.minimum(gt_y2, pred_y2)

    # 计算交集的面积
    inter_w, inter_h = (inter_x2 - inter_x1).clamp(min=0), (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 计算并集的面积
    gt_area, pred_area = gt_w * gt_h, pred_w * pred_h
    union_area = pred_area + gt_area - inter_area

    return inter_area / (union_area + 1e-6)  # [batch, s, s]

def calc_kmeans_iou(pred_box, center_box):
    import paddle

    d = []  # 每个pred_box到每个center_box的距离

    # 计算交集的面积, pred_box: [box_num, 2（w, h）]
    pred_box_area = pred_box[..., 0:1] * pred_box[..., 1:2]  # [pred_box_num]
    center_box_area = center_box[..., 0:1] * center_box[..., 1:2]  # [center_box_num]

    for i in range(pred_box.shape[0]):
        # 计算交集的面积
        min_width, min_height = paddle.minimum(pred_box[i, 0:1], center_box[..., 0:1]), paddle.minimum(pred_box[i, 1:2], center_box[..., 1:2])
        inter_area = min_width * min_height  # [center_box_num]

        # 计算并集的面积
        union_area = pred_box_area[i] + center_box_area - inter_area  # [center_box_num]

        d.append(inter_area / (union_area + 1e-6))  # [center_box_num]

    return paddle.to_tensor(d).squeeze()  # [pred_box_num, center_box_num]

# K-Means聚类生成预测框
def kmeans_anchor_boxes(boxes, k):
    import paddle
    # 从boxes中随机选k个边界框作为初始中心框
    center_boxes = boxes[paddle.randperm(boxes.shape[0])[:k]]

    # 每个边界框分配的中心框索引（初始值为0）, [box_num]
    box_center_box_idx = paddle.zeros(boxes.shape[0])

    while True:
        # 计算每个边界框与所有中心框的IOU
        ious = calc_kmeans_iou(boxes, center_boxes)

        # 找到每个边界框对应IOU最大的中心框索引（即分配到最近的中心框）
        box_max_center_box_idx = paddle.argmax(ious, axis=1)

        # 如果聚类分配结果不再变化（收敛状态），退出循环
        if paddle.all(box_max_center_box_idx == box_center_box_idx):
            # 按面积从小到大排序中心框
            return paddle.to_tensor(sorted(center_boxes, key=lambda area: area[0] * area[1], reverse=False))

        else:  # 更新中心框（未收敛状态）
            center_boxes = paddle.zeros_like(center_boxes)  # 重新初始化中心框

            for i in range(k):
                # 获取属于当前第i个聚类的所有边界框
                cls_i_boxes = boxes[paddle.where(box_max_center_box_idx == i)]
                # 计算属于当前第i个聚类的所有边界框的（width、height）的均值，作为新的中心框
                center_boxes[i] = paddle.mean(cls_i_boxes, axis=0)

            # 更新上一次的聚类分配结果
            box_center_box_idx = box_max_center_box_idx.copy()