from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(outputs, masks, num_classes, class_names):
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    correct_pixels = 0
    total_pixels = masks.numel()


    for cls in range(num_classes):
        ## IOU
        true_positive = ((outputs == cls) & (masks == cls)).sum().item()
        false_positive = ((outputs == cls) & (masks != cls)).sum().item()
        false_negative = ((outputs != cls) & (masks == cls)).sum().item()

        union = true_positive + false_positive + false_negative
        if union > 0:
            iou_per_class.append(true_positive / union)
        else:
            iou_per_class.append(0.0)

        ## Precision, Recall, F1 score
        pred_binary = (outputs == cls).flatten().cpu().numpy()
        mask_binary = (masks == cls).flatten().cpu().numpy()

        precision = precision_score(mask_binary, pred_binary, zero_division=0)
        recall = recall_score(mask_binary, pred_binary, zero_division=0)
        f1 = f1_score(mask_binary, pred_binary, zero_division=0)

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    # Pixel Accuracy
    correct_pixels = (outputs == masks).sum().item()
    mean_iou = sum(iou_per_class) / num_classes
    pixel_accuracy = correct_pixels / total_pixels

    for i, iou in enumerate(iou_per_class):
        print(f"Class {class_names[i]}: IoU={iou_per_class[i]:.4f}, Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}, F1-Score={f1_per_class[i]:.4f}")

    print(f"\nOverall Metrics: Mean IoU={mean_iou:.4f}, Pixel Accuracy={pixel_accuracy:.4f}")

    return mean_iou, pixel_accuracy, precision_per_class, recall_per_class, f1_per_class