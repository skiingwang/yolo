from __future__ import annotations
import time, numpy as np, paddle, matplotlib.pyplot as plt
from tqdm import tqdm
from utils import read_yaml, calc_iou, get_box, convert_real_xywh, nms
from yolov1 import YoloLoss

device = 'cuda' if paddle.device.is_compiled_with_cuda() else 'cpu'
MODEL_CONFIG = read_yaml('./config.yaml')


def train(model, train_loader, val_loader, epochs=10, lr=0.0001):
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=lr)
    loss_fn = YoloLoss()
    losses, val_losses = [], []
    since = time.time()
    for epoch in range(epochs):
        loop = tqdm(iterable=enumerate(train_loader), desc=f'Epoch [{epoch+1} / {epochs}]', total=len(train_loader), leave=True, ncols=150, dynamic_ncols=True, bar_format='{l_bar}{bar:15}{r_bar}')
        model = model.to(device).train()
        batch_losses = []
        for batch_idx, batch in loop:
            img, labels = batch
            img = img.to(device)
            logits = model(img)
            batch_loss = loss_fn(logits, labels)
            batch_losses.append(batch_loss)
            optimizer.clear_grad()
            batch_loss.backward()
            optimizer.step()
            loop.set_postfix(batch_loss=batch_loss.item())
        losses.append(np.mean(batch_losses))
        print(f'Epoch Avg Loss: {losses[-1]:.4f}')
        if val_loader is not None:
            mAP = evaluate(model, val_loader, device=device)
            print(f'Epoch Valid mAP: {mAP:.4f}')
        print('-'*32)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    return losses, val_losses


@paddle.no_grad()
def evaluate(model, dataloader, device, s=MODEL_CONFIG['MODEL']['SPLIT_SIZE'], conf_thresh=MODEL_CONFIG['INFERENCE']['CONFIDENCE_THRESHOLD'], iou_thresh=MODEL_CONFIG['INFERENCE']['IOU_THRESHOLD']):
    model.eval()

    total_AP = []

    for img, labels in dataloader:  # 遍历每个批次
        batch_size = img.shape[0]  # img.shape: [batch, 3, height, width]
        img = img.to(device)
        preds = model(img)  # [batch, 7, 7, 20]

        for b in range(batch_size):  # 遍历每个图像
            pred_logits, label = preds[b], labels[b]  # 获取每个图像的预测结果和标签值，[7, 7, 20]

            # --- 提取预测框和真实框的box和cls值 ---
            pred_confs1, pred_confs2 = paddle.nn.functional.sigmoid(pred_logits[..., 0]), paddle.nn.functional.sigmoid(pred_logits[..., 5])  # 获取预测置信度
            pred_boxes1_grid, pred_boxes2_grid, pred_cls = get_box(pred_logits)  # 获取预测框和类别
            pred_cls = paddle.nn.functional.sigmoid(pred_cls)

            gt_obj_mask = label[..., 0] == 1  # 获取置信度为1的物体网格索引矩阵，[7, 7] -> bool
            gt_rows, gt_cols = paddle.where(gt_obj_mask)  # 获取所有标注物体的行索引和列索引
            gt_boxes1_grid, gt_boxes2_grid, gt_cls = get_box(label)  # 获取标注框和类别，[7, 7, 4], [7, 7, 10]

            # --- 获取所有物体最佳预测框和置信度 ---
            iou1, iou2 = calc_iou(pred_boxes1_grid, gt_boxes1_grid), calc_iou(pred_boxes2_grid, gt_boxes1_grid)  # [7, 7]
            use_box1 = (iou1 >= iou2).astype('float32')  # 比较掩码，[7, 7] -> 0.0/1.0
            best_boxes = pred_boxes1_grid * use_box1.unsqueeze(-1) + pred_boxes2_grid * (1 - use_box1).unsqueeze(-1)  # 获取所有物体最佳预测框, [7, 7, 4]
            best_confs = pred_confs1 * use_box1 + pred_confs2 * (1 - use_box1)  # 获取所有物体最佳置信度

            #  --- 转换 Prediction ---
            pred_list = []
            for i in range(s):  # 遍历所有网格
                for j in range(s):
                    conf = best_confs[i, j].item()
                    if conf > 0:
                        cls = paddle.argmax(pred_cls[i, j]).item()
                        score = conf * pred_cls[i, j][cls].item()
                        if score > conf_thresh:  # 满足置信度阈值
                            ax, ay, aw, ah = convert_real_xywh(best_boxes[i, j].tolist(), i, j, s, img.shape[2], img.shape[3])
                            pred_list.append([ax, ay, aw, ah, cls, score])  # 有物体的网格的预测框（转换后）和类别
            current_preds = []  # 保存经过nms后的预测结果
            if len(pred_list) > 0:
                pred_tensor = paddle.to_tensor(pred_list)
                keep_idx = nms(pred_tensor[..., :4], pred_tensor[..., -1], iou_thresh)
                if len(keep_idx) > 0:
                    nms_preds = pred_tensor[keep_idx]
                    for p in nms_preds:
                        current_preds.append((int(p[4].item()), p[0].item(), p[1].item(), p[2].item(), p[3].item(), p[5].item()))  # {当前图像：预测类别和预测框}

            #  --- 转换 Ground Truth ---
            current_gts = []  # 保存所有标注物体
            for k in range(len(gt_rows)):  # 遍历所有标注物体
                row, col = gt_rows[k].item(), gt_cols[k].item()  # 获取当前标注物体的索引
                ax, ay, aw, ah = convert_real_xywh(gt_boxes1_grid[row, col].tolist(), row, col, s, img.shape[2], img.shape[3])  # 转换为实际值
                cls_id = paddle.where(gt_cls[row, col]==1)[0].item()  # 获取类别
                current_gts.append((cls_id, ax, ay, aw, ah))  # {当前图像：真实类别和标注框}

            TP, FP, FN = 0, 0, 0

            current_preds = sorted(current_preds, key=lambda x: x[5], reverse=True)
            matched_gt = [False] * len(current_gts)

            for pred in current_preds:
                best_iou, best_gt_idx = 0, -1
                pred_box = paddle.to_tensor(pred[1:5])
                for gt_idx, gt in enumerate(current_gts):
                    if pred[0] == gt[0] and not matched_gt[gt_idx]:
                        gt_box = paddle.to_tensor(gt[1:5])
                        iou = calc_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx  # 当前预测框最匹配的标注框索引
                if best_iou >= iou_thresh and best_gt_idx != -1:
                    matched_gt[best_gt_idx] = True  # 标记该标注框已被匹配
                    TP += 1
                else:
                    FP += 1  # 误报的预测框

            FN = sum(1 for m in matched_gt if not m)  # 漏报的标注框
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            total_AP.append(f1)

    mAP = paddle.mean(paddle.to_tensor(total_AP))
    model.train()
    return mAP


def figure(losses, val_losses, train_accs, val_accs):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses, 'ro-', label='Train Loss')
    plt.plot(val_losses, 'bo-', label='Valid Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'rs-',label='Train Acc')
    plt.plot(val_accs, 'bs-', label='Valid Acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()