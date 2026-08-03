import numpy as np
import torch
from skimage import measure
       
class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)  # TP, TP+FN
        inter, union = batch_intersection_union(preds, labels)  # TP, TP + FN + FP

        self.correct.append(correct)
        self.union.append(union)
        self.inter.append(inter)
        self.label.append(labeled)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

        self.cur_correct = correct
        self.cur_union = union
        self.cur_inter = inter
        self.cur_labeled = labeled

    def get_current(self):
        
        cur_pixAcc = 1.0 * self.cur_correct / (np.spacing(1) + self.cur_labeled)  # TP / (TP + FN)
        cur_IoU = 1.0 * self.cur_inter / (np.spacing(1) + self.cur_union)  # TP / (TP + FN + FP)
        return cur_pixAcc, cur_IoU

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)  # TP / (TP + FN)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)  # TP / (TP + FN + FP)
        mIoU = IoU.mean()
        return float(pixAcc), mIoU

    def get_single(self):
        self.correct = np.asarray(self.correct)
        self.union = np.asarray(self.union)
        self.inter = np.asarray(self.inter)
        self.label = np.asarray(self.label)

        spixAcc = 1.0 * self.correct / (np.spacing(1) + self.label)  # TP / (TP + FN)
        sIoU = 1.0 * self.inter / (np.spacing(1) + self.union)  # TP / (TP + FN + FP)
        nIoU = sIoU.mean()
        return spixAcc, nIoU
    
    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

        self.inter = []
        self.union = []
        self.correct = []
        self.label = []

        self.cur_inter = 0
        self.cur_union = 0
        self.cur_correct = 0
        self.cur_labeled = 0


class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0

        self.single_dismatch_pixel = []
        self.single_pixel = []
        self.single_PD = []
        self.single_target = []

        self.cur_dismatch_pixel = 0
        self.cur_all_pixel = 0
        self.cur_PD = 0
        self.cur_target = 0

    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64') 

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)   # number of targets / connected components
        self.single_target.append(len(coord_label))
        self.cur_target = len(coord_label)

        self.image_area_total = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        true_img = np.zeros(predits.shape)
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                    del coord_image[m]
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

        self.single_dismatch_pixel.append((predits - true_img).sum())
        self.single_pixel.append(int(size[0])*int(size[1]))
        self.single_PD.append(len(self.distance_match))

        self.cur_dismatch_pixel = (predits - true_img).sum()
        self.cur_all_pixel = int(size[0]) * int(size[1])
        self.cur_PD = len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target   # number of targets
        return Final_PD, float(Final_FA.cpu().detach().numpy())
    
    def get_current(self):
        cur_FA =  self.cur_dismatch_pixel / self.cur_all_pixel
        cur_PD =  self.cur_PD /self.cur_target   # number of targets
        return cur_PD, float(cur_FA)

    def get_single(self):
        self.single_dismatch_pixel = np.asarray(self.single_dismatch_pixel)
        self.single_target = np.asarray(self.single_target)
        self.single_pixel = np.asarray(self.single_pixel)
        self.single_PD = np.asarray(self.single_PD)

        single_FA = self.single_dismatch_pixel / self.single_pixel
        single_PD = self.single_PD / self.single_target
        return single_PD, single_FA

    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()  # TP + FN, all positive pixels in gt
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()  # TP, all true positive pixels
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())  # TP

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))  # TP
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))  # TP + FP
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))  # TP + FN
    area_union     = area_pred + area_lab - area_inter # TP + FP + FN 

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
