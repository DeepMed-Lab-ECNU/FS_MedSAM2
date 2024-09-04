import os
import numpy as np
import pdb

def normalize_img(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def npz_loader(npz_file):
    npz_data = np.load(npz_file)
    sup_img_part = npz_data['sup_img_part']
    sup_fgm_part = npz_data['sup_fgm_part']
    sup_bgm_part = npz_data['sup_bgm_part']
    query_images = npz_data['query_images']
    query_labels = npz_data['query_labels']
    labels_id = npz_data['labels']
    n_scan = npz_data['n_scan']

    sup_img_part = normalize_img(sup_img_part)
    query_images = normalize_img(query_images)
    sup_que_num = [sup_img_part.shape[0], query_images.shape[0]]
    images_np = np.concatenate([sup_img_part, query_images], axis=0)

    return [images_np, sup_fgm_part.squeeze(), sup_bgm_part, query_labels.squeeze(), labels_id, n_scan, sup_que_num]

from collections import defaultdict

def load_sup_img_part(npz_file):
    try:
        data = np.load(npz_file)
        return data['sup_img_part']
    except KeyError:
        print(f"{npz_file} does not contain 'sup_img_part'")
        return None

def extract_case_and_slice(file_name):
    parts = file_name.split('_')
    case_id = parts[3]  # 'case_2'
    slice_num = int(parts[-1].replace('.npz', '').replace('z', ''))
    return case_id, slice_num

def extract_case_slice_label(file_name):
    parts = file_name.split('_')
    label = int(parts[0])
    case_id = parts[3]  # 'case_2' 
    slice_num = int(parts[-1].replace('.npz', '').replace('z', ''))
    return case_id, slice_num, label

def group_sup_img_parts(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    grouped_files = []
    
    for file in files:
        npz_path = os.path.join(folder_path, file)
        sup_img_part = load_sup_img_part(npz_path)
        
        if sup_img_part is not None:
            found_group = False
            for group in grouped_files:
                if np.array_equal(group['sup_img_part'], sup_img_part):
                    group['files'].append(file)
                    found_group = True
                    break
            
            if not found_group:
                grouped_files.append({'sup_img_part': sup_img_part, 'files': [file]})

    detailed_groups = []
    
    for group in grouped_files:
        case_dict = defaultdict(list)
        
        for file in group['files']:
            case_id, slice_num, label_id = extract_case_slice_label(file)
            case_dict[f"{case_id}_{label_id}"].append((slice_num, file))
        
        for case_label, slices in case_dict.items():
            slices.sort()  
            file_list = [file for _, file in slices]
            detailed_groups.append({'sup_img_part': group['sup_img_part'], 'case_id': case_label, 'files': file_list})
    
    return detailed_groups
