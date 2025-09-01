import torch

def get_entropy_of_dataset(data_tensor: torch.Tensor):
    labels_list = [row[-1].item() for row in data_tensor]  
    
    def entropy_calc(probabilities):
        return -torch.sum(probabilities * torch.log2(probabilities))
    
    unique_labels = list(set(labels_list))
    
    probs = []
    for label in unique_labels:
        count = labels_list.count(label)
        probs.append(count / len(labels_list))
    
    entropy_val = entropy_calc(torch.tensor(probs, dtype=torch.float32)).item()
    return entropy_val


def get_avg_info_of_attribute(data_tensor: torch.Tensor, attr_index: int):
    column_vals = [row[attr_index].item() for row in data_tensor]
    unique_attr_vals = list(set(column_vals))
    labels = [row[-1].item() for row in data_tensor]
    
    total = len(column_vals)
    avg_info_val = 0.0

    for val in unique_attr_vals:
        freq = column_vals.count(val)
        prob_val = freq / total
        
        tmp_attr, tmp_labels = [], []
        for idx in range(len(column_vals)):
            if column_vals[idx] == val:
                tmp_attr.append(val)
                tmp_labels.append(labels[idx])
        
        subset_tensor = torch.cat((
            torch.tensor(tmp_attr).unsqueeze(1), 
            torch.tensor(tmp_labels).unsqueeze(1)
        ), dim=1)
        
        subset_entropy = get_entropy_of_dataset(subset_tensor)
        
        if not torch.isnan(torch.tensor(subset_entropy)):
            avg_info_val += prob_val * subset_entropy
    
    return avg_info_val


def get_information_gain(data_tensor: torch.Tensor, attr_index: int):
    dataset_entropy = get_entropy_of_dataset(data_tensor)
    avg_info = get_avg_info_of_attribute(data_tensor, attr_index)
    info_gain = torch.round(torch.tensor(dataset_entropy - avg_info), decimals=4).item()
    return info_gain


def get_selected_attribute(data_tensor: torch.Tensor):
    gain_map = {}
    for i in range(len(data_tensor[0]) - 1):
        gain_map[i] = get_information_gain(data_tensor, i)
    
    max_gain_val = max(gain_map.values())
    
    for k, v in gain_map.items():
        if v == max_gain_val:
            return gain_map, k
    
    return {}, -1
