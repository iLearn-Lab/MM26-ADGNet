from utils import *
import os
import os.path as osp
import json
from PIL import Image
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DataSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, mode, img_norm_cfg=None):
        super().__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode in ('val', 'test'):
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        self.tokenizer = CLIPTokenizer.from_pretrained('./clip-vit-base-patch16')
        
        # ==========================================
        # Pre-load foreground and background text captions
        # Select JSON files based on mode (train / val / test)
        # ==========================================
        self.captions_fg = {}
        self.captions_bg = {}
        
        if mode == 'train':
            fg_paths = [osp.join(dataset_dir, 'text', 'train_fg.json')]
            bg_paths = [osp.join(dataset_dir, 'text', 'train_bg.json')]
        elif mode in ('val', 'test'):
            fg_paths = [osp.join(dataset_dir, 'text', 'test_fg.json')]
            bg_paths = [osp.join(dataset_dir, 'text', 'test_bg.json')]
        
        # Load foreground / target descriptions
        for path in fg_paths:
            if osp.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.captions_fg.update(json.load(f))
            else:
                print(f"Warning: FG caption file not found: {path}")

        # Load background / scene descriptions
        for path in bg_paths:
            if osp.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.captions_bg.update(json.load(f))
            else:
                print(f"Warning: BG caption file not found: {path}")
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', file_name + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', file_name + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', file_name + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', file_name + '.bmp'))

        img = img.resize((256, 256), Image.NEAREST)
        mask = mask.resize((256, 256), Image.NEAREST)

        # ==========================================
        # Get foreground and background captions for this sample
        # ==========================================
        text_fg = self.captions_fg[file_name]
        text_bg = self.captions_bg[file_name]

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = img, mask
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)

            img_patch = img_patch[np.newaxis, :]
            mask_patch = mask_patch[np.newaxis, :]

            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            inputs_fg = self.tokenizer(text_fg, padding='max_length', max_length=20,truncation=True, return_tensors="pt")
            text_ids_fg = inputs_fg.input_ids      
            l_mask_fg = inputs_fg.attention_mask

            inputs_bg = self.tokenizer(text_bg, padding='max_length', max_length=20,truncation=True, return_tensors="pt")
            text_ids_bg = inputs_bg.input_ids      
            l_mask_bg = inputs_bg.attention_mask

            return img_patch, mask_patch, text_ids_fg, l_mask_fg, text_ids_bg, l_mask_bg
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))

            inputs_fg = self.tokenizer(text_fg, padding='max_length', max_length=20,truncation=True, return_tensors="pt")
            text_ids_fg = inputs_fg.input_ids
            l_mask_fg = inputs_fg.attention_mask

            inputs_bg = self.tokenizer(text_bg, padding='max_length', max_length=20,truncation=True, return_tensors="pt")
            text_ids_bg = inputs_bg.input_ids   
            l_mask_bg = inputs_bg.attention_mask
            
            return img, mask, [h, w], file_name, text_ids_fg, l_mask_fg, text_ids_bg, l_mask_bg
    
    def __len__(self):
        return len(self.files)


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]

        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
            
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            
        return input, target