import paddle, cv2, numpy as np, matplotlib.pyplot as plt
from utils import get_box, convert_real_xywh, nms

# 假设你的配置和辅助函数都在上下文中可用
device = 'cuda' if paddle.device.is_compiled_with_cuda() else 'cpu'

# 1. 类别名称映射 (根据你的数据集修改，假设是VOC的20类)
CLASSES = ['cola', 'pepsi', 'sprite', 'fanta', 'spring', 'ice', 'scream', 'milk', 'red', 'king']

def preprocess(img_path, input_size=448):
    """图像预处理：缩放、转置、归一化、增加Batch维度"""
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"图像加载失败: {img_path}")

    orig_img = img.copy()  # 保存原图用于最后画框
    h, w = orig_img.shape[:2]

    # 缩放到模型输入尺寸 (448x448)
    img = cv2.resize(img, (input_size, input_size))
    # BGR 转 RGB (因为通常训练时用的是RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 归一化到 [0, 1]，如果你的训练时有做均值方差归一化，这里也要保持一致
    img = img / 255.0

    # (H, W, C) -> (C, H, W)
    img = img.transpose((2, 0, 1))
    # 增加批次维度 -> (1, C, H, W)
    img = np.expand_dims(img, axis=0).astype('float32')

    return paddle.to_tensor(img), orig_img, (h, w)


def predict_single_image(model, img_path, s=7, conf_thresh=0.25, iou_thresh=0.5):
    """对单张图像进行推理并可视化"""
    # 1. 加载模型权重并设置为评估模式
    # load_weight(model, './yolo1.pdparams')  # 假设你已经加载过了
    model.eval()

    # 2. 图像预处理
    img_tensor, orig_img, (orig_h, orig_w) = preprocess(img_path, input_size=448)
    img_tensor = img_tensor.to(device)

    # 3. 模型前向推理
    with paddle.no_grad():  # 推理时关闭梯度计算，节省显存
        preds = model(img_tensor)  # 输出形状: [1, 7, 7, 30] (假设2框20类)

    # 4. 提取预测结果 (因为只有一张图，取第0个batch)
    pred_logits = preds[0]  # [7, 7, 30]

    # --- 以下后处理逻辑与你 evaluate 函数中的一致 ---
    pred_confs1, pred_confs2 = pred_logits[..., 0], pred_logits[..., 5]
    pred_boxes1_grid, pred_boxes2_grid, pred_cls = get_box(pred_logits)

    # 获取最佳框
    # 注意：这里 calc_iou 是预测框与真实框的IoU，但在推理时没有真实框。
    # YOLOv1原版逻辑是：选择两个框中置信度更高的那个，而不是用IoU去选！
    # 修正：推理时应使用置信度比较，而非IoU比较
    use_box1 = (pred_confs1 >= pred_confs2).astype('float32')
    best_boxes = pred_boxes1_grid * use_box1.unsqueeze(-1) + pred_boxes2_grid * (1 - use_box1).unsqueeze(-1)
    best_confs = pred_confs1 * use_box1 + pred_confs2 * (1 - use_box1)

    pred_list = []
    # 遍历网格
    for i in range(s):
        for j in range(s):
            conf = best_confs[i, j].item()
            if conf > 0:  # 只要有置信度就提取类别
                cls = paddle.argmax(pred_cls[i, j]).item()
                score = conf * pred_cls[i, j][cls].item()
                if score > conf_thresh:
                    # 转换为448x448图像上的真实坐标
                    ax, ay, aw, ah = convert_real_xywh(best_boxes[i, j].tolist(), i, j, s, 448, 448)
                    pred_list.append([ax, ay, aw, ah, cls, score])

    # 5. 执行 NMS (非极大值抑制)
    final_boxes = []
    if len(pred_list) > 0:
        pred_tensor = paddle.to_tensor(pred_list)
        keep_indices = nms(pred_tensor[:, :4], pred_tensor[:, -1], iou_thresh)

        if keep_indices.shape[0] > 0:
            nms_preds = pred_tensor[keep_indices]
            # 6. 将坐标从 448x448 映射回原始图像尺寸
            for p in nms_preds:
                px, py, pw, ph = p[0].item(), p[1].item(), p[2].item(), p[3].item()
                cls_id = int(p[4].item())
                score = p[5].item()

                # 映射回原图比例
                # YOLO输出的 x,y 是中心点，w,h 是宽高
                ratio_w = orig_w / 448.0
                ratio_h = orig_h / 448.0

                orig_cx = px * ratio_w
                orig_cy = py * ratio_h
                orig_w_box = pw * ratio_w
                orig_h_box = ph * ratio_h

                # 转换为左上角和右下角坐标 (x1, y1, x2, y2) 用于OpenCV画框
                x1 = int(orig_cx - orig_w_box / 2)
                y1 = int(orig_cy - orig_h_box / 2)
                x2 = int(orig_cx + orig_w_box / 2)
                y2 = int(orig_cy + orig_h_box / 2)

                # 边界裁剪，防止超出图片范围
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w - 1, x2), min(orig_h - 1, y2)

                final_boxes.append((cls_id, score, x1, y1, x2, y2))

    # 7. 可视化绘制
    for cls_id, score, x1, y1, x2, y2 in final_boxes:
        # 画框
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 写类别和分数
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        # 计算文本背景大小
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(orig_img, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 显示图像 (因为OpenCV是BGR，matplotlib是RGB，需要转换一下显示才正常)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(orig_img_rgb)
    plt.axis("off")
    plt.show()

    return final_boxes
