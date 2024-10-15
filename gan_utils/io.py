import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import tables as tb
import torch
from .functions import charge2img, normalize_and_convert_to_img

class StereoVeristasDataGenNorm(Dataset):
    def __init__(self, input_file, size_threshold=100, imsize=96):
        self.h5file = input_file
        self.size_threshold = size_threshold
        self.img_size = imsize
        self.h5_table = tb.open_file(self.h5file)
        
        self.T1 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_001.cols.image)
        self.T1_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_001.cols.event_id)
        
        self.T2 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_002.cols.image)
        self.T2_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_002.cols.event_id)
        
        self.T3 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_003.cols.image)
        self.T3_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_003.cols.event_id)
        
        self.T4 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_004.cols.image)
        self.T4_events = np.array(self.h5_table.root.dl1.event.telescope.parameters.tel_004.cols.event_id)
        
        self.T1_cond = np.where(np.sum(self.T1, axis=1)>= size_threshold)
        self.T2_cond = np.where(np.sum(self.T2, axis=1)>= size_threshold)
        self.T3_cond = np.where(np.sum(self.T3, axis=1)>= size_threshold)
        self.T4_cond = np.where(np.sum(self.T4, axis=1)>= size_threshold)
        
        self.t1_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events[self.T1_cond], self.T2_events), self.T3_events), self.T4_events)
        self.t2_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events[self.T2_cond]), self.T3_events), self.T4_events)
        self.t3_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events[self.T3_cond]), self.T4_events)
        self.t4_intersect_events = np.intersect1d(np.intersect1d(np.intersect1d(self.T1_events, self.T2_events), self.T3_events), self.T4_events[self.T4_cond])
        
        self.all_intersected_events = np.unique(np.concatenate([self.t1_intersect_events, self.t2_intersect_events, self.t3_intersect_events, self.t4_intersect_events]))
        print(f'Found total samples {self.all_intersected_events.shape[0]} meeting size thresh {self.size_threshold}')
        
    def __len__(self):
        return(len(self.all_intersected_events))
    
    def __getitem__(self, index):
        
        intersect_event = self.all_intersected_events[index]
        
        t1_array = normalize_and_convert_to_img(self.T1[np.where(self.T1_events == intersect_event)][0])
        t2_array = normalize_and_convert_to_img(self.T2[np.where(self.T2_events == intersect_event)][0])
        t3_array = normalize_and_convert_to_img(self.T3[np.where(self.T3_events == intersect_event)][0])
        t4_array = normalize_and_convert_to_img(self.T4[np.where(self.T4_events == intersect_event)][0])
        
        return torch.cat((t1_array,t2_array,t3_array,t4_array), dim=0)
    
    

class VeritasDataGen(Dataset):
    def __init__(self, input_file, size_threshold=100, imsize=96):
        self.h5file = input_file
        self.size_threshold = size_threshold
        self.img_size = imsize
        self.h5_table = tb.open_file(self.h5file)
        
        self.T1 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_001.cols.image)
        self.T1_sum = np.sum(self.T1, axis=1)
        self.T1_cond = self.T1[np.where(self.T1_sum>= size_threshold)]
        
        self.T2 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_002.cols.image)
        self.T2_sum = np.sum(self.T2, axis=1)
        self.T2_cond = self.T2[np.where(self.T2_sum>= size_threshold)]
        
        self.T3 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_003.cols.image)
        self.T3_sum = np.sum(self.T3, axis=1)
        self.T3_cond = self.T3[np.where(self.T3_sum>= size_threshold)]
        
        self.T4 = np.array(self.h5_table.root.dl1.event.telescope.images.tel_004.cols.image)
        self.T4_sum = np.sum(self.T4, axis=1)
        self.T4_cond = self.T4[np.where(self.T4_sum>= size_threshold)]
        
        self.Tall = np.concatenate([self.T1_cond, self.T2_cond, self.T3_cond, self.T4_cond], axis=0)
        print(f'Found total samples {self.Tall.shape[0]} meeting size thresh {self.size_threshold}')
        
    def __len__(self):
        return(len(self.Tall))
    
    
    def __getitem__(self, index, info_key='image'):
        
        working_array = self.Tall[index]
        
        working_array = working_array/np.percentile(working_array[np.where(working_array!=0)],95)
        working_array[(np.where(np.array(working_array)<0))[0]] = 0
        
#         print(working_array.shape)
        return torch.unsqueeze(torch.Tensor(charge2img(working_array, imsize=self.img_size).astype('float32')), dim=0)
    
    
class VeritasDataGenNorm(VeritasDataGen):
    def __getitem__(self, index):
        working_array = self.Tall[index]
        working_array = working_array/np.sum(working_array)
        working_array[(np.where(np.array(working_array)<0))[0]] = 0
        
        percentile_value = np.percentile(working_array, 99)
        working_array[np.where(working_array>percentile_value)] = percentile_value
        
        working_array = working_array/np.max(working_array)
        return torch.unsqueeze(torch.Tensor(charge2img(working_array, imsize=self.img_size).astype('float32')), dim=0)
    
    
        