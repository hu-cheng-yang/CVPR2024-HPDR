from misc.utils import AverageMeter
from misc.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def countMetric(prob_list, label_list):
    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold*100


def eval(valid_dataloader, model, prot, isLoop=True, isProt=False):
    prob_dict = {}
    label_dict = {}

    model.eval()
    prot.eval()

    print('testing, the len of data_loader_target is', len(valid_dataloader))
    stop_iter = 50

    with torch.no_grad():
        for iter, (input, target, videoID, subtype) in enumerate(valid_dataloader):
            if iter % 50 == 0:
                print(iter)
            if iter==stop_iter and isLoop:
                break
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

            _, _, feat = model(input)
            cls_out = prot.forward_cls(feat)

            prob = F.softmax(cls_out, dim=-1)[:, 1].cpu().detach().numpy()
            label = target.cpu().data.numpy()
            try:
                videoID = videoID.cpu().data.numpy()
            except:
                videoID = videoID
            for i in range(len(prob)):
                if(videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)

    print('len(label_list)', len(label_list))
    print('len(prob_list)', len(prob_list))
    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)
    return [cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold*100]

