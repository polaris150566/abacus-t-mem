import torch
import torch.nn.functional as F


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def pad_to_length(data, max_len, dim=0,value=0):
    shape = [d for d in data.shape] 
    assert(shape[dim] <= max_len)
    if shape[dim] <= max_len:
        return data

    pad_len = max_len-shape[dim]
    pad_shape = [[0, 0]] * dim + [[0, pad_len]] + [[0, 0]] * (len(shape) - dim -1) 
    data_pad = F.pad(data,convert_pad_shape(pad_shape),mode='constant', value=value) 
    return data_pad

def pad_to_length_2d(data, max_len1, max_len2, dim1=0, dim2=1,value=0):
    shape = [d for d in data.shape] 
    assert(shape[dim1] <= max_len1) 
    assert(shape[dim2] <= max_len2)
    if shape[dim1] == max_len1 and shape[dim2] == max_len2:
        return data
    pad_len1 = max_len1- shape[dim1] 
    pad_len2 = max_len2- shape[dim2]
    pad_shape = []
    for i in range(len(shape)):
        if i == dim1:
            pad_shape.append([0,pad_len1]) 
        elif i == dim2:
            pad_shape.append([0,pad_len2]) 
        else:
            pad_shape. append([0, 0])

    data_pad =F.pad(data,convert_pad_shape(pad_shape),mode= 'constant', value=value) 
    return data_pad

def pad_and_stack(data_list, dim=0,value=0):
    max_len = max([d.shape[dim] for d in data_list]) 
    data = torch.stack(
        [pad_to_length(d, max_len, dim, value) for d in data_list], dim =0)
    return data

def pad_and_stack_2d(data_list, dim1=0, dim2=1,value=0):
    max_len1 = max([d.shape[dim1] for d in data_list] ) 
    max_len2 = max([d.shape[dim2] for d in data_list]) 
    data = torch.stack(
        [pad_to_length_2d(d,max_len1, max_len2, dim1, dim2, value) for d in data_list], 
        dim =0)
    return data