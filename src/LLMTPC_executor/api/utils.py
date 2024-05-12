from typing import List, Set, Dict
from collections import Counter
# from shapely.geometry import Polygon, LineString
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append('../')
sys.path.append('../src/LLMTPC_executor')
from api.spatial_relation_helper import HorizontalProximityHelper, VerticalProximityHelper, BetweenHelper, AllocentricHelper
import numpy as np
import math
import random
from dataset.data_utils import export
from scipy.spatial.transform import Rotation as R

horizontal_relation2clock = {
    # 'front': [10, 11, 12, 1, 2],
    'front right': [1, 2],
    'right front': [1, 2],
    # 'right': [1, 2, 3, 4, 5],
    'back right': [4, 5],
    'behind right': [4, 5],
    'right behind': [4, 5],
    # 'back': [4, 5, 6, 7, 8],
    'back left': [7, 8],
    'behind left': [7, 8],
    'left behind': [7, 8],
    # 'left': [7, 8, 9, 10, 11],
    'front left': [10, 11],
    'left front': [10, 11]
}
horizontal_clock = {
    "1 o'clock": [12, 1, 2],
    "2 o'clock": [1, 2, 3],
    "3 o'clock": [2, 3, 4],
    "4 o'clock": [3, 4, 5],
    "5 o'clock": [4, 5, 6],
    "6 o'clock": [5, 6, 7],
    "7 o'clock": [6, 7, 8],
    "8 o'clock": [7, 8, 9],
    "9 o'clock": [8, 9, 10],
    "10 o'clock": [9, 10, 11],
    "11 o'clock": [10, 11, 12],
    "12 o'clock": [11, 12, 1]
}
attribute_candidates = {
    'color':["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "silver", "white", "yellow"],# "beige", "gold", 'checkered','patterned',],
    'shape':["c-shaped", "circle", "circular", "in rows", "l-shaped", "oval", "rectangle", "rectangular", "round", "square", "triangle", "u-shaped"],
    'material':["ceramic", "glass", "leather", "metal", "wood"],

    'size':['large','queen','small','twin'],
    'length':['long','short'],
    'pattern': ['checkered','patterned','solid'],
    'openness': ['closed','open','partially open','partially closed'],
    'cleanliness': ['messy','tidy'],
    'on_off': ['off','on'],
    'occupancy': ['full','empty'],
}

attribute_synonym = {
    "white": ["white", "silver", "grey", "beige"],
    "silver": ["silver", "grey", "white", "beige"],
    "grey": ["grey", "silver", "white", "beige"],
    "yellow": ["yellow", "beige", "gold"],
    "brown": ["brown", "beige", "gold"],
    "beige": ["beige", "yellow", "brown", "gold", "white"],
    "gold": ["brown", "yellow", "beige"],


    "circle": ["circle", "circular", "oval", "round"],
    "circular": ["circle", "circular", "oval", "round"],
    "oval": ["circle", "circular", "oval", "round"],
    "round": ["circle", "circular", "oval", "round"],
    "rectangle": ["rectangle", "rectangular"],
    "rectangular": ["rectangular", "rectangle"],

    "wood": ["wood", "wooden"]
}


def _transform_object_ref(obj_pos: List[float], ref_pose: List[float]) -> List[float]:
    # obj_pos: xyz; agent_pose: xyz,quaternion
    # transform object position to agent coordinate
    _agent_pos = ref_pose[0:3]
    # move
    transform_matrix = np.array([[1, 0, 0, -_agent_pos[0]],
                [0, 1, 0, -_agent_pos[1]],
                [0, 0, 1, -_agent_pos[2]],
                [0, 0, 0, 1]])
    obj_pos = np.append(obj_pos, 1)
    obj_pos = np.dot(obj_pos, transform_matrix.T)
    # rotate
    if len(ref_pose) > 3:
        _agent_quat = ref_pose[3:7]
        r = R.from_quat(_agent_quat)
        angles = r.as_euler('xyz', degrees=False)
        angle = 3/2*np.pi - angles[2]
        rotation_matrix = np.array([[np.cos(angle), -np.sin(rotation_matrix), 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        obj_pos = np.dot(obj_pos, rotation_matrix.T)
    return obj_pos

def _rel_pos_to_clock(rel_pos: List[float]) -> int:
    angle = np.arctan2(rel_pos[1], rel_pos[0])
    angle_to_y = (90 - math.degrees(angle)) % 360
    if angle_to_y >= 360-15 or angle_to_y < 15:
        return 12
    for hour, slot in enumerate(range(30, 360-29, 30)):
        if slot-15 <= angle_to_y < slot+15:
            return hour+1

        
class ThreeDObject():
    def __init__(self, object_id, point_clouds, label, rgbs, bbox):
        self.object_id = object_id # -1 means agent
        self.point_clouds = point_clouds
        self.label = label
        self.rgbs = rgbs
        self.bbox = bbox # cx, cy, cz, lx, ly, lz
        self.extrema = self._extrema()
        self.corners = self._corners()
    
    def _extrema(self):
        cx, cy, cz, lx, ly, lz = self.bbox
        xmin = cx - lx / 2.0
        xmax = cx + lx / 2.0
        ymin = cy - ly / 2.0
        ymax = cy + ly / 2.0
        zmin = cz - lz / 2.0
        zmax = cz + lz / 2.0
        return np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    
    def _corners(self):
        xmin, ymin, zmin, xmax, ymax, zmax = self.extrema
        corners = np.array([[xmin, ymin, zmin],
                            [xmin, ymax, zmin],
                            [xmax, ymin, zmin],
                            [xmax, ymax, zmin],
                            [xmin, ymin, zmax],
                            [xmin, ymax, zmax],
                            [xmax, ymin, zmax],
                            [xmax, ymax, zmax]])
        return corners
    
    def _z_face(self):
        corners = self.corners
        [_, _, zmin, _, _, zmax] = self.extrema
        bottom_face = corners[np.array(corners[:, 2], np.float32) == np.array(zmin, np.float32), :]
        return bottom_face

    def iou_2d(self, other):
        a = self.corners
        b = other.corners
        
        a_xmin, a_xmax = np.min(a[:, 0]), np.max(a[:, 0])
        a_ymin, a_ymax = np.min(a[:, 1]), np.max(a[:, 1])

        b_xmin, b_xmax = np.min(b[:, 0]), np.max(b[:, 0])
        b_ymin, b_ymax = np.min(b[:, 1]), np.max(b[:, 1])

        box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
        box_b = [b_xmin, b_ymin, b_xmax, b_ymax]

        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])

        # compute the area of intersection rectangle
        inter_area = max(0, xB - xA) * max(0, yB - yA)

        box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

        iou = inter_area / float(box_a_area + box_b_area - inter_area)
        i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
        a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]
        return iou, i_ratios, a_ratios
    
    def intersection(self, other, axis=2):
        l_min, l_max = self.extrema[axis], self.extrema[axis + 3]

        other_l_min, other_l_max = other.extrema[axis], other.extrema[axis + 3]

        a = max(l_min, other_l_min)
        b = min(l_max, other_l_max)
        i = b - a

        return i, i / (l_max - l_min), i / (other_l_max - other_l_min)

    def distance_from_other_object(self, other):
        nn = NearestNeighbors(n_neighbors=1).fit(self.point_clouds)
        distances, _ = nn.kneighbors(other.point_clouds)
        res = np.min(distances)
        return res

class ObjectAttribute:
    def __init__(self,
            object_id: int,
            category: str,
            xyz: List[float],
            lwh: List[float],
            rgb: List[int],
            direction: int,
            caption: str
        ):
        self.object_id = object_id
        self.category = category
        self.xyz = xyz
        self.lwh = lwh
        self.rgb = rgb
        self.direction = direction
        self.caption = caption
        self.distance = None

class Scene:
    def __init__(self, object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, question, answers, caption) -> None:
        self.object_pcds = object_pcds  # nxNx6, [[[x,y,z,r,g,b], ...], ...]]
        self.object_labels = object_labels  # ["table", ...], only include objects whose nyu40id is in OBJ_CLASS_IDS
        self.object_ids = object_ids    # object_id of selected_object_label in raw *.aggregation.json(0-indexed)
        self.instance_bboxes = instance_bboxes  # [[x,y,z,l,w,h], ...]
        self.instance_rgbs = instance_rgbs  # [[r,g,b], ...]
        self.situation = situation  # str
        self.question = question    # str
        self.answers = answers  # [str]
        self.caption = caption  # {object_id: [str]}
        self.object_id_to_label = {object_id: label for object_id, label in zip(self.object_ids, self.object_labels)}
    
    def get_object_set(self) -> List[ObjectAttribute]:
        object_set = set()
        for i in range(len(self.object_labels)):
            caption = f"This is a {self.object_labels[i]}. No caption for this {self.object_labels[i]}. Try other methods or use your commonsense reasoning skills to infer missing information about the {self.object_labels[i]}."
            if self.object_ids[i] in self.caption:
                caption = " ".join(self.caption[self.object_ids[i]])
            object_set.add(ObjectAttribute(
                object_id=self.object_ids[i],
                category=self.object_labels[i],
                xyz=self.instance_bboxes[i][0:3],
                lwh = self.instance_bboxes[i][3:6],
                rgb=self.instance_rgbs[i],
                direction=_rel_pos_to_clock(self.instance_bboxes[i][0:3]),
                caption=caption
            ))
        return object_set
    
    def get_context(self, num_captions=30) -> str:
        if len(self.caption) > num_captions:
            object_ids = random.sample(list(self.caption.keys()), k=num_captions)
        else:
            object_ids = list(self.caption.keys())
        context = []
        def count_words(sentence):
            words = sentence.split()
            return len(words)

        for object_id in object_ids:
            context.append((self.object_id_to_label[object_id], max(self.caption[object_id], key=count_words)))
        context.sort()
        context = " ".join([f"{x[0]}: {x[1]}" for x in context])
        return context

    def get_unique_category(self) -> List[str]:
        return sorted(list(set(self.object_labels)))
    
    def get_object_num(self):
        object_num = Counter(self.object_labels)
        object_num = {item: count for item, count in sorted(object_num.items())}
        return object_num
    
    def get_object(self, object_id):
        idx = np.argwhere(np.array(self.object_ids) == object_id)[0][0]
        return ThreeDObject(
            object_id=object_id,
            point_clouds=self.object_pcds[idx][:, :3],
            label=self.object_labels[idx],
            rgbs=self.object_pcds[idx][:, 3:],
            bbox=self.instance_bboxes[idx][:6]
        )
    