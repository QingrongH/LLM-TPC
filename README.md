# LLM-TPC
Code for the paper "Think-Program-reCtify: 3D Situated Reasoning with Large Language Models"

[[Project Page](https://qingrongh.github.io/LLM-TPC/)] [[Paper](https://arxiv.org/abs/2404.14705)]
<img src="docs/assets/LLM-TPC.png"/>

## Install
```Shell
conda create -n llm-tpc python=3.9 -y
conda activate llm-tpc
pip install openai==0.28 numpy scikit-learn matplotlib omegaconf torch torch_redstone einops tqdm open_clip_torch trimesh plyfile shapely
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
```

## Dataset
Organize the data as follows in `data`.
```Shell
data
├── openshape
│   ├── model.pt
│   └── open_clip_pytorch_model.bin
├── qa
│   └── SQA_test.json
├── scans
│   ├── scene0000_00
│   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
│   │   ├── scene0000_00_vh_clean_2.labels.ply
│   │   ├── scene0000_00_vh_clean_2.ply
│   │   ├── scene0000_00.aggregation.json
│   │   └── scene0000_00.txt
│   └── ...
└── scannetv2-labels.combined.tsv
```

### ScanNet
To acquire the access to ScanNet dataset, please refer to [ScanNet](https://github.com/ScanNet/ScanNet) and follow the instructions there. You will get a `download-scannet.py` script after your request for the ScanNet dataset is approved. Use the commands below to download the portion of ScanNet that is necessary for LLM-TPC:
```Shell
python download-scannet.py -o data --type _vh_clean_2.0.010000.segs.json
python download-scannet.py -o data --type _vh_clean_2.labels.ply
python download-scannet.py -o data --type _vh_clean_2.ply
python download-scannet.py -o data --type .aggregation.json
python download-scannet.py -o data --type .txt
```

### SQA3D
Download the [question-answer pairs](https://zenodo.org/record/7792397/files/ScanQA_format.zip) from [SQA3D](https://github.com/SilongYong/SQA3D) and put `SQA_test.json` under `data/qa`.

### OpenShape
We use the [pointbert-vitg14-rgb](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/tree/main) and [OpenCLIP ViT-bigG-14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main) checkpoint from [OpenShape](https://github.com/Colin97/OpenShape_code).
Download `model.pt` from [here](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb/tree/main) and `open_clip_pytorch_model.bin` from [here](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/tree/main). Put them under `data/openshape`.

## Inference
TODO
```Shell
cd scripts
# Input your OPENAI_API_KEY in 'no_caption-openshape-gt_seg-gt_label/config.json'
python example.py --agent no_caption-openshape-gt_seg-gt_label/config.json
```

## Evaluation
TODO
```Shell
cd scripts
python eval.py --log_dir ../logs/test/no_caption-openshape-gt_seg-gt_label
```

## Visualization
TODO
```Shell
cd src/dataset
python visualize_bbox.py
```

## Acknowledgement
- [Agents](https://github.com/aiwaves-cn/agents): the codebase we built upon.
- [ReferIt3D](https://github.com/referit3d/referit3d): we design APIs for spacial relation recognition based on ReferIt3D.
- [OpenShape](https://github.com/Colin97/OpenShape_code): we design APIs for open-vocabulary object attribute classification based on OpenShape.
- [ScanRefer](https://github.com/daveredrum/ScanRefer): code for visualization.


## Citation:
```
@article{qingrong2024llm-tpc,
  title={Think-Program-reCtify: 3D Situated Reasoning with Large Language Models},
  author={Qingrong He and Kejun Lin and Shizhe Chen and Anwen Hu and Qin Jin},
  journal={arXiv preprint arXiv:2404.14705},
  year={2024}
}
```