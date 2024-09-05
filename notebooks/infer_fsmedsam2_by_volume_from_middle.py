import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb
from utils.npz_loader import npz_loader, group_sup_img_parts, normalize_img
import sys
from PIL import Image
import cv2
saved_npz_path = '/path/to/saved_npz'
ckpt_path = '/path/to/ckpt'

def calculate_dice_compo(prediction, groundtruth):

    tp = np.sum((prediction == 1) & (groundtruth == 1))
    fp = np.sum((prediction == 1) & (groundtruth == 0))
    fn = np.sum((prediction == 0) & (groundtruth == 1))
    return tp, fp, fn


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.build_fsmedsam2 import build_fsmedsam2_video_predictor


data_dict = {}
for i, group in enumerate(group_sup_img_parts(saved_npz_path)):
    exp_data = np.load(os.path.join(saved_npz_path, group['files'][0]))
    case_id = group['case_id']
    if case_id not in data_dict:
        data_dict[case_id] = []
    sup_img_part = exp_data['sup_img_part']
    sup_img_part = normalize_img(sup_img_part)
    sup_fgm_part = exp_data['sup_fgm_part']
    label_id = exp_data['labels'][0]

    infer_nums = len(group['files'])
    query_imgs = []
    query_labels = []
    query_names = group['files']

    for infer_id in range(infer_nums):
        cur_image = np.load(os.path.join(saved_npz_path, group['files'][infer_id]))['query_images']
        cur_image = normalize_img(cur_image)
        cur_label = np.load(os.path.join(saved_npz_path,group['files'][infer_id]))['query_labels']
        query_imgs.append(cur_image)
        query_labels.append(cur_label)

    data_dict[case_id].append({'sup_img': sup_img_part, 'sup_label': sup_fgm_part, 'label_id': label_id, 'query_imgs': query_imgs, 'query_labels': query_labels,'query_names':query_names})
    
model_type_dict = {'sam2_hiera_t':'sam2_hiera_tiny', 'sam2_hiera_s':'sam2_hiera_small', 'sam2_hiera_b+':'sam2_hiera_base_plus','sam2_hiera_l':'sam2_hiera_large'}
for k,v in model_type_dict.items():
    metric_dict = {}
    sam2_checkpoint = os.path.join(ckpt_path, f'{v}.pt')
    model_cfg = f"{k}.yaml"

    predictor = build_fsmedsam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    for case_id in data_dict:
        data_list = data_dict[case_id]
        sup_img_list = []
        sup_label_list = []
        query_imgs_list = []
        query_labels_list = []
        query_names_list = []
        label_id = data_list[0]['label_id']
        for sub_volume in data_list:
            sup_img_list.append(sub_volume['sup_img'])
            sup_label_list.append(sub_volume['sup_label'])
            query_imgs_list.extend(sub_volume['query_imgs'])
            query_labels_list.extend(sub_volume['query_labels'])
            query_names_list.extend(sub_volume['query_names'])
        combined = list(zip(query_imgs_list, query_labels_list, query_names_list))
        combined_sorted = sorted(combined, key=lambda x: int(x[2].split('_z')[1].split('.')[0]))
        query_imgs_list, query_labels_list, query_names_list = zip(*combined_sorted)
        query_imgs_list, query_labels_list, query_names_list = list(query_imgs_list), list(query_labels_list), list(query_names_list)
        sup_img_list, sup_label_list = [sup_img_list[1]], [sup_label_list[1]]
        left_idx = len(query_imgs_list)//2
        right_idx = left_idx + len(sup_img_list)
        
        all_images = np.concatenate(query_imgs_list[:left_idx]+sup_img_list+query_imgs_list[left_idx:],axis=0)
        all_labels = np.concatenate(query_labels_list[:left_idx]+sup_label_list+query_labels_list[left_idx:],axis=0)
        
        inference_state = predictor.init_state_by_np_data(images_np=all_images)
        predictor.reset_state(inference_state)

        labels = np.array([1], np.int32)

        for i in range(len(sup_img_list)):
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=i+left_idx,
                obj_id=1,
                mask=sup_label_list[i][0],
            )
        
        video_segments = np.zeros((all_images.shape[0], all_images.shape[-2], all_images.shape[-1]), dtype=np.uint8)


        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=right_idx):
            video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=left_idx-1, reverse=True):
            video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()

        all_vis = []
        for j in range(len(all_images)):
            if j >= left_idx and j< right_idx:
                continue
            query_pred = video_segments[j].squeeze()
            query_labels = all_labels[j].squeeze()
            # query_img = normalize_img(all_images[j])*255
            # query_img = query_img.transpose(1, 2, 0).astype(np.uint8)
            # vis_img1 = query_img.copy()
            # vis_img2 = query_img.copy()
            # vis_img3 = query_img.copy()

            # vis_img2[query_pred>0] = (0,255,255)
            # vis_img3[query_labels>0] = (255,0,255)
            # all_vis.append(np.hstack([vis_img1, vis_img2, vis_img3]))
            tp, fp, fn = calculate_dice_compo(query_pred, query_labels)
            
            if label_id not in metric_dict:
                metric_dict[label_id] = {"tp": 0, "fp": 0, "fn": 0}
            metric_dict[label_id]["tp"] += tp
            metric_dict[label_id]["fp"] += fp
            metric_dict[label_id]["fn"] += fn
        # concat_img = np.vstack(all_vis)
        # cv2.imwrite(f'{save_dir}/case_{case_id}.jpg', concat_img)

        # slice_dice = metric.dc(query_pred, query_labels)
        # if labels_id[0] not in metric_dict:
        #     metric_dict[labels_id[0]] = []
        # metric_dict[labels_id[0]].append(slice_dice)
    for k, v in metric_dict.items():
        print(model_cfg)
        print(f"label: {k}, dice: {2 * v['tp'] / (2 * v['tp'] + v['fp'] + v['fn'])}")
    del predictor

