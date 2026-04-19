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

# 计算交并比
def calc_iou(pred_box, gt_box):
    import paddle
    # gt_box: [s, s, 4（x, y, w, h）]
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

    return inter_area / (union_area + 1e-6)  # [s, s]
