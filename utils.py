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
    # gt_box: [batch, s, s, xywh）]
    gt_x, gt_y, gt_w, gt_h = gt_box[..., 0], gt_box[..., 1], gt_box[..., 2], gt_box[..., 3]
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
    inter_w, inter_h = paddle.clamp((inter_x2 - inter_x1), min=0), paddle.clamp((inter_y2 - inter_y1), min=0)
    inter_area = inter_w * inter_h

    # 计算并集的面积
    gt_area, pred_area = gt_w * gt_h, pred_w * pred_h
    union_area = pred_area + gt_area - inter_area

    return inter_area / (union_area + 1e-6)  # [batch, s, s]

# 获取框、类别
def get_box(targets):
    # [s, s, 2 * 5 + 10]
    box1, box2, cls = targets[..., 1:5], targets[..., 6:10], targets[..., 10:]
    return box1, box2, cls

# 将局部归一化值转换为实际值
def convert_real_xywh(boxes, row_idx, col_idx, s, img_width, img_height):
    x, y, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x, y = (x + col_idx) * (img_width / s), (y + row_idx) * (img_height / s)
    w, h = (w / s) * img_width, (h / s) * img_height
    return x, y, w, h

# 非极大值抑制
def nms(pred_boxes, pred_scores, iou_threshold):
    import paddle
    if len(pred_boxes) == 0:
        return paddle.to_tensor([], dtype='int64')

    order_scores = paddle.argsort(pred_scores, descending=True)  # 从大到小的索引序列， [n]

    px, py, pw, ph = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]  # [n]
    x1, y1 = px - pw / 2, py - ph / 2  # 框左上角xy坐标
    x2, y2 = px + pw / 2, py + ph / 2  # 框右下角xy坐标
    areas = pw * ph

    keep = []
    while len(order_scores) > 0:
        if len(order_scores) == 1:
            keep.append(order_scores[0].item())
            break
        i = order_scores[0].item()
        keep.append(i)

        # 计算当前得分最高的框与其余框的交并比掩码
        xx1, yy1 = paddle.maximum(x1[i], x1[order_scores[1:]]), paddle.maximum(y1[i], y1[order_scores[1:]])  # [len(order_scores[1:])]
        xx2, yy2 = paddle.minimum(x2[i], x2[order_scores[1:]]), paddle.minimum(y2[i], y2[order_scores[1:]])

        w, h = paddle.clip(xx2 - xx1, min=0.0), paddle.clip(yy2 - yy1, min=0.0)  # 过滤负值
        inter = w * h  # [len(order_scores[1:])]，即[n-1]

        iou = inter / (areas[i] + areas[order_scores[1:]] - inter + 1e-6)
        mask = iou <= iou_threshold  # 抑制重叠度高的框
        remain = paddle.where(mask)[0]  # mask为True的索引
        order_scores = order_scores[remain + 1]  # 还原真实索引

    return paddle.to_tensor(keep, dtype='int64')