import json
import os
import sys
sys.path.append("../src")
sys.path.append("../src/dataset")
from data_utils import export   #, export_pcd, load_meta_data, load_data, load_Mask3D200_data
from LLMTPC_executor.api.api import Scene
from tqdm import tqdm

class SceneDataset:
    def __init__(self, scene_info: dict) -> None:
        meta_info = scene_info['meta_info']
        split = meta_info['split']
        self.scene_dir = meta_info['scene_dir']
        self.label_map_file = meta_info['label_map_file']
        qa_file = meta_info['qa_file'].format(split)
        caption_file = meta_info["caption_file"]
        self.qa = self.prepare_qa(qa_file)
        self.caption = self.prepare_caption(caption_file)

        self.object_pcd_setting = scene_info['object_pcd_setting']
        
    
    @classmethod
    def from_config(cls, config):
        # with open(config_path) as f:
        #     config = json.load(f)
        return cls(config)
    
    def prepare_qa(self, qa_file):
        print("...prepare qa...")
        with open(qa_file) as f:
            data = json.load(f)
        qa = {}
        for item in tqdm(data):
            answers = item['answers']
            question = item['question']
            situation = item['situation']
            question_id = str(item['question_id'])
            scene_id = item['scene_id']
            position = item['position']
            if not scene_id in qa:
                qa[scene_id] = {}
            qa[scene_id][question_id] = {
                "question": question,
                "answers": answers,
                "situation": situation,
                "position": position
            }
        return qa
    
    def prepare_caption(self, caption_file):
        caption = {}
        if not os.path.exists(caption_file):
            for scene_id in self.qa:
                caption[scene_id] = {}
            return caption
        print("...prepare caption...")
        with open(caption_file) as f:
            data = json.load(f)
        caption = {}
        for scene_id, object in tqdm(data.items()):            
            caption[scene_id] = {}
            for object_id, cap in object.items():
                object_id = int(object_id)
                caption[scene_id][object_id] = []   
                for caption_id, content in cap.items():
                    caption[scene_id][object_id].append(content["description"])
        return caption

    def get_scene(self, scene_id, question_id):
        if self.object_pcd_setting["label_type"] != "GT":
            raise NotImplementedError
        else:
            return self.get_scene_gt(scene_id, question_id)

    def get_scene_gt(self, scene_id, question_id):
        base_scene_dir = os.path.join(self.scene_dir, scene_id)
        mesh_file = os.path.join(base_scene_dir, scene_id+"_vh_clean_2.ply")
        agg_file = os.path.join(base_scene_dir, scene_id+".aggregation.json")
        seg_file = os.path.join(base_scene_dir, scene_id+"_vh_clean_2.0.010000.segs.json")
        position = self.qa[scene_id][question_id]['position']
        object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs = export(mesh_file, agg_file, seg_file, self.label_map_file, position)
        situation = self.qa[scene_id][question_id]['situation']
        question = self.qa[scene_id][question_id]['question']
        answers = self.qa[scene_id][question_id]['answers']
        scene_caption = self.caption[scene_id]
        return Scene(object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, question, answers, scene_caption)

