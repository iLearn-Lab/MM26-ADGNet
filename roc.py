import os
import glob
import numpy as np
import cv2
from scipy.io import loadmat
from sklearn.metrics import auc
from tqdm import tqdm

# ==========================================
# 1. ROCMetric class
# ==========================================
class ROCMetric(object):
    def __init__(self, bins=10):
        self.bins = bins
        self.reset()

    def update(self, pred, label):
        # guard against division by zero
        max_val = np.max(pred)
        if max_val > 0:
            pred = pred / max_val  # normalize output to [0, 1]
        else:
            pred = np.zeros_like(pred)  # all-zero prediction
            
        label = label.astype(np.uint8)

        # analysis target number
        num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(label)
        
        # skip if no targets (background only)
        if(num_labels <= 1):
            return

        # get masks and update background area and targets number
        back_mask = labels == 0
        tmp_back_area = np.sum(back_mask)
        self.background_area += tmp_back_area
        self.target_nums += (num_labels - 1)

        for ibin in range(self.bins + 1):
            thre = ibin / self.bins
            pred_binary = pred >= thre

            # update false detection
            tmp_false_detect = np.sum(np.logical_and(back_mask, pred_binary))
            assert tmp_false_detect <= tmp_back_area
            self.false_detect[ibin] += tmp_false_detect

            # update true detection, there maybe multiple targets
            for t in range(1, num_labels):
                target_mask = labels == t
                self.true_detect[ibin] += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        # guard against zero denominator
        fpr = self.false_detect / max(self.background_area, 1e-6)  # X axis
        tpr = self.true_detect / max(self.target_nums, 1e-6)       # Y axis
        return fpr, tpr

    def get_all(self):
        return self.false_detect, self.background_area, self.true_detect, self.target_nums

    def reset(self):
        self.false_detect = np.zeros(self.bins+1)
        self.true_detect = np.zeros(self.bins+1)
        self.background_area = 0
        self.target_nums = 0

# ==========================================
# 2. Main evaluation loop
# ==========================================
def evaluate_roc(pred_dir, gt_dir, pred_ext='.mat', gt_ext='.png'):
    print(f"Loading predictions from: {pred_dir}")
    print(f"Loading ground truth from: {gt_dir}")
    
    roc_metric = ROCMetric(bins=10)
    
    # Find all prediction files and match to ground truth
    pred_files = glob.glob(os.path.join(pred_dir, f'*{pred_ext}'))
    if len(pred_files) == 0:
        raise ValueError("No prediction files found. Please check the pred_dir and extension.")
        
    for pred_path in tqdm(pred_files, desc="Evaluating"):
        filename_without_ext = os.path.splitext(os.path.basename(pred_path))[0]
        gt_path = os.path.join(gt_dir, filename_without_ext + gt_ext)
        
        # check if corresponding ground truth file exists
        if not os.path.exists(gt_path):
            print(f"\nWarning: Ground truth file missing for {filename_without_ext}, skipping...")
            continue
            
        # read ground truth
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            continue
        gt_mask = (gt_img > 0).astype(np.uint8) 
        
        # read prediction
        if pred_ext == '.mat':
            mat_data = loadmat(pred_path)
            if 'predict_map' in mat_data:
                pred_img = mat_data['predict_map']
            else:
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                pred_img = mat_data[keys[0]]
        else:
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred_img = pred_img.astype(np.float32) / 255.0

        if gt_mask.shape != pred_img.shape:
            gt_mask = cv2.resize(gt_mask, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        roc_metric.update(pred_img, gt_mask)

    fpr, tpr = roc_metric.get()
    roc_auc = auc(fpr, tpr)
    print(f"\nEvaluation Complete! AUC: {roc_auc:.4f}")
    
    return fpr, tpr, roc_auc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="ROC evaluation from saved .mat predictions")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing .mat prediction files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth mask .png files')
    parser.add_argument('--pred_ext', type=str, default='.mat', help='Prediction file extension')
    parser.add_argument('--gt_ext', type=str, default='.png', help='Ground truth file extension')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: parent of pred_dir)')
    args = parser.parse_args()

    FPR, TPR, roc_auc = evaluate_roc(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        pred_ext=args.pred_ext,
        gt_ext=args.gt_ext
    )

    print("-" * 50)
    print(f"AUC: {roc_auc:.4f}")
    print(f"TPR Array ({len(TPR)} bins):")
    print(TPR)
    print(f"FPR Array ({len(FPR)} bins):")
    print(FPR)

    # Save to txt in output directory
    output_dir = args.output_dir or os.path.dirname(os.path.normpath(args.pred_dir))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'roc_metrics.txt')
    with open(save_path, 'w') as f:
        f.write(f"AUC: {roc_auc:.4f}\n")
        f.write(f"TPR ({len(TPR)} bins): {', '.join(f'{v:.6f}' for v in TPR)}\n")
        f.write(f"FPR ({len(FPR)} bins): {', '.join(f'{v:.10f}' for v in FPR)}\n")
    print(f"Saved to {save_path}")