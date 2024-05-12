import os
import glob
import h5py
import numpy as np
from sklearn import metrics
import os
import numpy as np
import open3d as o3d
import random
import torch
import sys
from param import parse_args
import models
import MinkowskiEngine as ME
from utils.data import normalize_pc
from utils.misc import load_config
from huggingface_hub import hf_hub_download
from collections import OrderedDict, defaultdict
import open_clip
import re
from PIL import Image
import torch.nn.functional as F

texts = ["red"]

openclip_cache_dir="/data7/lkj/code/OpenShape_code"

open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', cache_dir=openclip_cache_dir)
open_clip_model.cuda().eval()

text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
feat = open_clip_model.encode_text(text_tokens)
print()