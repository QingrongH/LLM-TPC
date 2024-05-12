import os
import numpy as np
# import open3d as o3d
import random
import torch
import sys
sys.path.append('../src/LLMTPC_executor/api/openshape')
sys.path.append('api/openshape')
from torch.nn.parallel import DataParallel
from param import parse_args
import models
# import MinkowskiEngine as ME
from .utils.data import normalize_pc
from .utils.misc import load_config
from huggingface_hub import hf_hub_download
from collections import OrderedDict, defaultdict
import open_clip
import re
from PIL import Image
import torch.nn.functional as F

class Match_SQA3D_Attr():
    def __init__(self,
                 config_path='OpenShape_code/src/configs/train.yaml',
                 model_name='OpenShape/openshape-pointbert-vitg14-rgb',
                 openclip_model_path="laion2b_s39b_b160k",
                 openshape_model_path="cache/huggingface/hub/models--OpenShape--openshape-pointbert-vitg14-rgb/snapshots/d771a992218de10b967eea690e8f474aa96dd758/model.pt",
                 openclip_cache_dir="OpenShape_code"
                 ):
        self.has_init_model = False
        self.config_path = config_path
        self.model_name = model_name
        self.openclip_model_path = openclip_model_path
        self.openshape_model_path = openshape_model_path
        self.openclip_cache_dir = openclip_cache_dir
        # self.openclip_device = 'cuda:1'
        self.openclip_device = 'cuda'


    def init_model(self):
        print("loading OpenShape model...")
        cli_args, extras = parse_args(sys.argv[1:])
        self.config = load_config(self.config_path, cli_args = vars(cli_args), extra_args = extras)
        self.model = self.load_model(self.config, self.model_name)
        # self.model = DataParallel(self.model)
        self.model.eval()
        print("loading OpenCLIP model...")
        # self.open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir=self.openclip_cache_dir)
        self.open_clip_model, _ = open_clip.create_model_from_pretrained(
            'ViT-bigG-14',
            pretrained=self.openclip_model_path
        )
        self.open_clip_model.to(self.openclip_device).eval()
        self.has_init_model = True

    def normalize_pc(self, pc):
        # normalize pc to [-1, 1]
        pc = pc - np.mean(pc, axis=0)
        if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
            pc = np.zeros_like(pc)
        else:
            pc = pc / np.max(np.linalg.norm(pc, axis=1))
        return pc
    
    def process_data(self, xyz, rgb, num_points=10000):
        n = xyz.shape[0]
        if n > num_points:
            idx = random.sample(range(n), num_points)
            xyz = xyz[idx]
            rgb = rgb[idx]
        elif n < num_points:
            xyz = np.concatenate([xyz, xyz[np.random.randint(n, size=[10000 - n])]], axis=0)
            rgb = np.concatenate([rgb, rgb[np.random.randint(n, size=[10000 - n])]], axis=0)
        xyz = self.normalize_pc(xyz)
        if rgb is None:
            rgb = np.ones_like(rgb) * 0.4
        rgb = rgb / 255.0 # normalize rgb to [0, 1]
        features = np.concatenate([xyz, rgb], axis=1)
        xyz = torch.from_numpy(xyz).type(torch.float32)
        features = torch.from_numpy(features).type(torch.float32)
        return xyz, features

    def load_model(self, config, model_name="OpenShape/openshape-spconv-all"):
        model = models.make(config).cuda()

        # if config.model.name.startswith('Mink'):
        #     model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) # minkowski only
        # else:
        #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
        checkpoint = torch.load(self.openshape_model_path)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in checkpoint['state_dict'].items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
        model.load_state_dict(model_dict)
        return model

    @torch.no_grad()
    def extract_text_feat(self, texts, clip_model,):
        text_tokens = open_clip.tokenizer.tokenize(texts).to(self.openclip_device)
        return clip_model.encode_text(text_tokens)

    def match(self, pcd, rgb, text):
        ''' predicts a single object
        @param pcd: np array of shape (N, 3)
        @param rgb: np array of shape (N, 3)
        @param text: str
        '''
        if not self.has_init_model:
            self.init_model()
        xyz, feat = self.process_data(pcd, rgb)
        xyz, feat = xyz.unsqueeze(0).cuda(), feat.unsqueeze(0).cuda()
        shape_feat = self.model(xyz, feat, device='cuda', quantization_size=self.config.model.voxel_size).detach().cpu()
        text_feat = self.extract_text_feat([text], self.open_clip_model).detach().cpu()
        similarity = F.normalize(shape_feat, dim=1) @ F.normalize(text_feat, dim=1).T
        if similarity[0] > 0.1:
            return True
        else:
            return False

    def classify(self, pcd, rgb, texts):
        ''' classifies a single object
        @param text: list of str
        '''
        if not self.has_init_model:
            self.init_model()
        xyz, feat = self.process_data(pcd, rgb)
        xyz, feat = xyz.unsqueeze(0).cuda(), feat.unsqueeze(0).cuda()
        shape_feat = self.model(xyz, feat, device='cuda', quantization_size=self.config.model.voxel_size).detach().cpu()
        text_feat = self.extract_text_feat(texts, self.open_clip_model).detach().cpu()
        similarity = F.normalize(shape_feat, dim=1) @ F.normalize(text_feat, dim=1).T
        pred_class = texts[similarity.argmax().item()]

        # similarity_scores = []
        # for i in range(len(texts)):
        #     similarity_scores.append((texts[i], similarity[:, i].item()))
        # sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # for score in sorted_scores:
        #     print(score[0], score[1])
        
        return pred_class
        
def main():
    match_sqa3d_attr = Match_SQA3D_Attr()
    match_sqa3d_attr.match(np.random.randn(10000, 3), np.random.randn(10000, 3), "a")
    # eval_scannet.visualize()

if __name__=='__main__':
    main()