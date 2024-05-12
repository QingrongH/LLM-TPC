import os
from typing import Set, Dict, List, Union, Optional
import sys
sys.path.append('../src/LLMTPC_executor')
from api.utils import (
    ObjectAttribute, Scene, ThreeDObject,
    HorizontalProximityHelper, VerticalProximityHelper, BetweenHelper, AllocentricHelper,
    horizontal_clock, horizontal_relation2clock, attribute_candidates, attribute_synonym,
    _transform_object_ref, _rel_pos_to_clock, export
)
from api.openshape.match_sqa3d_attr import Match_SQA3D_Attr
import numpy as np
np.argmax
def generate_array(N, w1, w2, w3):
    column1 = np.random.uniform(-w1, w1, size=(N, 1))
    column2 = np.random.uniform(-w2, w2, size=(N, 1))
    column3 = np.random.uniform(-w3, w3, size=(N, 1))
    array = np.hstack((column1, column2, column3))
    means = np.mean(array, axis=0)
    array -= means
    return array
agent_pcd = generate_array(100, 0.3/2, 0.3/2, 0.3/2)

global use_caption
global use_openshape
global agent
agent = ThreeDObject(-1, agent_pcd, 'agent', np.array([0,0,0]), np.array([0,0,0,0.3, 0.3, 0.3]))
use_caption = True
use_openshape = True

global HpHelper, VpHelper, BtHelper, AlHelper

HpHelper = HorizontalProximityHelper(enable_gap=False)
VpHelper = VerticalProximityHelper()
BtHelper = BetweenHelper()
AlHelper = AllocentricHelper(positive_occ_thresh=0.7)

AlHelper_query_relation = AllocentricHelper(positive_occ_thresh=0.8)

# global match_sqa3d_attr

def scene() -> Set[ObjectAttribute]:
    """
    Returns a set of objects in the scene.
    """
    return scene_data.get_object_set()

def filter(object_set: Set[ObjectAttribute], category: str) -> Set[ObjectAttribute]:
    """
    Returns a set of objects whose category is category.

    Parameters
    ----------
    object_set : A set of objects in the scene.
    category : The category of objects to be filtered.
    """
    error_msg = 'Calling filter(object_set: Set[ObjectAttribute], category: str) -> Set[ObjectAttribute] error!'
    if isinstance(category, List):
        ret_object_set = set()
        for cat in category:
            assert isinstance(cat, str), f'{error_msg} type of each element of `category` must be str, not {type(cat)}.'
            assert cat in scene_data.get_unique_category(), f'{error_msg} each element of `category` must be chosen from {scene_data.get_unique_category()}, not "{cat}". There is no {category} in the room.'
            ret_object_set.update(set([item for item in object_set if item.category == cat]))
        return ret_object_set
    assert isinstance(category, str), f'{error_msg} type of `category` must be str, not {type(category)}.' 
    assert category in scene_data.get_unique_category(), f'{error_msg} `category` must be chosen from {scene_data.get_unique_category()}, not "{category}". There is no {category} in the room.'
    return set([item for item in object_set if item.category == category])

def relate(object_set: Set[ObjectAttribute], reference_object: ObjectAttribute, relation: str, is_anchor_agent=False) -> Set:
    """
    Returns a set of objects that are related to the reference_object by the relation.

    Parameters
    ----------
    object_set : A set of objects in the scene.
    reference_object : The reference object.
    relation : The relation between the object and the reference_object. Must be chosen from the following list: 
        ["above", "on", "below", "under", "left", "right", "front", "back", "behind", "closest", "farthest", "within reach", "touch", "around", 
        "front right", "back right", "right behind", "front left", "back left", "left behind",
        "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock", "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock"]
    """
    legal_relations = ["on", "above", "on top", "on top of", "below", "under", "below/under", "left", "right", "front", "back", "behind", "back/behind", "closest", "farthest", "far", "far away", "within reach", "beside", "at", "next", "next to", "touch", "around", "in"]
    legal_relations.extend(list(horizontal_clock.keys()))
    legal_relations.extend(list(horizontal_relation2clock.keys()))
    legal_relations.extend(["one o'clock", "two o'clock", "three o'clock", "four o'clock", "five o'clock", "six o'clock", "seven o'clock", "eight o'clock", "nine o'clock", "ten o'clock", "eleven o'clock", "twelve o'clock"])
    show_legal_relations = '["on", "above", "below", "under", "left", "right", "front", "back", "closest", "farthest", "within reach", "front right", "back right", "front left", "back left", "1 o\'clock", "2 o\'clock", ..., "12 o\'clock"]'
    error_msg = 'Calling relate(object_set: Set[ObjectAttribute], reference_object: ObjectAttribute, relation: str) -> Set error!'
    assert isinstance(object_set, Set) or isinstance(object_set, filter), f'{error_msg} type of `object_set` must be Set[ObjectAttribute], not {type(object_set)}.'
    if not is_anchor_agent:
        if isinstance(reference_object, Set) and reference_object != object_set:
            ret_obj = set()
            for ref_obj in reference_object:
                ret_obj.update(relate(object_set, ref_obj, relation))
            return ret_obj
        assert isinstance(reference_object, ObjectAttribute), f'{error_msg} type of `reference_object` must be ObjectAttribute, not {type(reference_object)}.'
    assert isinstance(relation, str), f'{error_msg} type of relation must be str, not {type(relation)}.' 
    assert relation in legal_relations, \
        f'{error_msg} `relation` must be chosen from {show_legal_relations}, not {relation}.'
    
    anchor_object = agent if is_anchor_agent else scene_data.get_object(reference_object.object_id)
    
    output_object_set = set()
    # agent_pose = scene_data.agent_pose
    if relation in ['closest', 'farthest', 'furthest', "far", "far away"]:
        if relation == "furthest":
            relation = relation.replace('furthest', 'farthest')
        elif relation == "far":
            relation = relation.replace('far', 'farthest')
        elif relation == "far away":
            relation = relation.replace('far away', 'farthest')
        _object_list = [scene_data.get_object(i.object_id) for i in object_set]
        ret_hp_cf = HpHelper.infer_close_far(_object_list, anchor_object)
        return set([obj for obj in object_set if obj.object_id == ret_hp_cf[relation]])

    for obj in object_set:
        if obj == reference_object:
            continue
        ref_xyz = anchor_object.bbox[:3] if is_anchor_agent else reference_object.xyz
        rel_pos = _transform_object_ref(obj.xyz, ref_xyz)
        target_object = scene_data.get_object(obj.object_id)
        if relation in ['above', 'on', "on top", "on top of", 'below', 'under', "below/under", "in"]:
            relation = relation.replace('under', 'below')
            relation = relation.replace('below/under', 'below')
            relation = relation.replace("in", "on")
            relation = relation.replace("on top", "on")
            relation = relation.replace("on top of", "on")
            ret_vp = VpHelper.infer(target_object, anchor_object)

            if relation in ret_vp:
                output_object_set.add(obj)
        elif relation in ['within reach', 'next',  'next to','beside', 'at', 'touch', 'around']:
            relation = relation.replace('touch', 'within reach')
            relation = relation.replace('at', 'within reach')
            relation = relation.replace('beside', 'within reach')
            relation = relation.replace('next to', 'within reach')
            relation = relation.replace('next', 'within reach')
            ret_hp_ta = HpHelper.infer_touch_around(target_object, anchor_object)
            if relation in ret_hp_ta:
                output_object_set.add(obj)
        elif relation in ['left', 'right', 'front', 'back', 'behind', "back/behind"]:
            relation = relation.replace("back/behind", "back")
            relation = relation.replace('behind', 'back')
            ret_al = AlHelper.infer(target_object, anchor_object)
            if relation in ret_al:
                output_object_set.add(obj)
        elif relation in ['left front', 'front left', 'right front', 'front right', 'left back', 'back left', 'left behind', 'behind left', 'right back', 'back right', 'right behind', 'behind right']:
            relation = relation.replace('behind', 'back')
            relation = relation.replace('back left', 'left back')
            relation = relation.replace('back right', 'right back')
            relation = relation.split()
            ret_al = AlHelper.infer(target_object, anchor_object)
            if relation[0] in ret_al and relation[1] in ret_al:
                output_object_set.add(obj)
        elif relation in list(horizontal_relation2clock.keys()):
            if _rel_pos_to_clock(rel_pos) in horizontal_relation2clock[relation]:
                output_object_set.add(obj)
        elif relation in list(horizontal_clock.keys()):
            clock = horizontal_clock[relation]
            if obj.direction in clock:
                output_object_set.add(obj)
    return output_object_set

def relate_agent(object_set: Set[ObjectAttribute], relation: str) -> Set:
    """
    Returns a set of objects that are related to the agent(you) by the relation.

    Parameters
    ----------
    object_set : A set of objects in the scene.
    relation : The relation between the object and the agent(you).
    """
    legal_relations = ["on", "above", "on top", "on top of", "below", "under", "below/under", "left", "right", "front", "back", "behind", "back/behind", "closest", "farthest", "far", "far away", "within reach", "next", "next to", "beside", "at", "touch", "around", "in"]
    show_legal_relations = '["on", "above", "below", "under", "left", "right", "front", "back", "closest", "farthest", "within reach", "front right", "back right", "front left", "back left", "1 o\'clock", "2 o\'clock", ..., "12 o\'clock"]'
    legal_relations.extend(list(horizontal_clock.keys()))
    legal_relations.extend(list(horizontal_relation2clock.keys()))
    legal_relations.extend(["one o'clock", "two o'clock", "three o'clock", "four o'clock", "five o'clock", "six o'clock", "seven o'clock", "eight o'clock", "nine o'clock", "ten o'clock", "eleven o'clock", "twelve o'clock"])
    error_msg = 'Calling relate_agent(object_set: Set[ObjectAttribute], relation: str) -> Set error!'
    assert isinstance(object_set, Set) or isinstance(object_set, filter), f'{error_msg} type of `object_set` must be Set[ObjectAttribute], not {type(object_set)}.'
    assert isinstance(relation, str), f'{error_msg} type of `relation` must be str, not {type(relation)}.' 
    assert relation in legal_relations, f'{error_msg} `relation` must be chosen from {show_legal_relations}, not {relation}.'
    return relate(object_set, None, relation, is_anchor_agent=True)

def relate_ternary(object_set: Set[ObjectAttribute], reference_object_a: ObjectAttribute, reference_object_b: ObjectAttribute, relation: str) -> Set:
    """
    Returns a set of objects that are related to the reference_object_a and reference_object_b by the relation.

    Parameters
    ----------
    object_set : A set of objects in the scene.
    reference_object_a : The first reference object.
    reference_object_b : The second reference object.
    relation : The relation between the object and the reference_object_a and reference_object_b. Must be chosen from the following list: 
        ["between"]
    """
    legal_relations = ["between"]
    error_msg = 'Calling relate_ternary(object_set: Set[ObjectAttribute], reference_object_a: ObjectAttribute, reference_object_b: ObjectAttribute, relation: str) -> Set error!'
    assert isinstance(object_set, Set), f'{error_msg} type of `object_set` must be Set[ObjectAttribute], not {type(object_set)}.'
    assert isinstance(reference_object_a, ObjectAttribute), f'{error_msg} type of `reference_object_a` must be ObjectAttribute, not {type(reference_object_a)}.'
    assert isinstance(reference_object_b, ObjectAttribute), f'{error_msg} type of `reference_object_b` must be ObjectAttribute, not {type(reference_object_b)}.'
    assert relation in legal_relations, f'{error_msg} `relation` must be chosen from {legal_relations}, not {relation}.'
    anchor_object_a = scene_data.get_object(reference_object_a.object_id)
    anchor_object_b = scene_data.get_object(reference_object_b.object_id)
    output_object_set = set()
    for object in object_set:
        target_object = scene_data.get_object(object.object_id)
        ret_bt = BtHelper.infer(target_object, anchor_object_a, anchor_object_b)
        if ret_bt:
            output_object_set.add(object)
    return output_object_set

def query_relation(object: ObjectAttribute, reference_object: ObjectAttribute, candidate_relations: List[str]=None, is_anchor_agent=False) -> Dict:
    """
    Returns a dict of vertical and allcentric relations between the object and the reference_object.

    Parameters
    ----------
    object : The object.
    reference_object : The reference object.
    """
    error_msg = 'Calling query_relation(object: ObjectAttribute, reference_object: ObjectAttribute) error!'
    if candidate_relations:
        error_msg = 'Calling query_relation(object: ObjectAttribute, reference_object: ObjectAttribute, candidate_relations: List[str]) error!'
        if isinstance(candidate_relations, str):
            candidate_relations = [candidate_relations]
        assert isinstance(candidate_relations, List), f'{error_msg} type of `candidate_relations` must be List, not {type(candidate_relations)}.'
    
    if not is_anchor_agent:
        assert isinstance(reference_object, ObjectAttribute), f'{error_msg} type of `reference_object` must be ObjectAttribute, not {type(reference_object)}.'
    else:
        assert reference_object is None, f'{error_msg} query_relation() takes 2 positional argument but 3 were given.'
    
    if isinstance(object, Set):
        ret_rel = set()
        for obj in object:
            ret_rel.update(query_relation(obj, reference_object, candidate_relations, is_anchor_agent))
        return list(ret_rel)
    
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    
    ret_relations = []
    anchor_object = agent if is_anchor_agent else scene_data.get_object(reference_object.object_id)
    target_object = scene_data.get_object(object.object_id)
    ret_al = AlHelper_query_relation.infer(target_object, anchor_object) # list[str]
    ret_relations.extend(ret_al)
    if candidate_relations:
        chosen_ret_relations = []
        for candidate_relation in candidate_relations:
            if candidate_relation in ret_relations:
                chosen_ret_relations.append(candidate_relation)
            elif candidate_relation in ["backward", "backwards", "behind", "behind me", "at my back"] and "back" in ret_relations:
                chosen_ret_relations.append(candidate_relation)
            elif candidate_relation in ["forward", "forwards", "in front", "infront", "in front of", "in front of me"] and "front" in ret_relations:
                chosen_ret_relations.append(candidate_relation)
            elif is_anchor_agent and "clock" in candidate_relation:
                chosen_ret_relations.append(str(object.direction)+" o'clock")
        if len(chosen_ret_relations)>0:
            return chosen_ret_relations
    return ret_relations

def query_relation_agent(object: ObjectAttribute, candidate_relations: List[str]=None) -> Dict:
    """
    Returns a dict of vertical, allcentric, horizontal and clock direction relations between the object and the agent(you).

    Parameters
    ----------
    object : The object.

    Examples
    --------
    >>> relation = query_relation_agent(chair)
    >>> print(relation)
    {'vertical': ['below'], 'allcentric': ['left'], 'horizontal': ['within reach'], 'clock': 9}
    # The chair is below the agent(you), on the left of the agent(you), within reach and in the direction of 9 o'clock from the agent(you).
    """
    error_msg = 'Calling query_relation_agent(object: ObjectAttribute) error!'
    if candidate_relations:
        error_msg = 'Calling query_relation_agent(object: ObjectAttribute, candidate_relations: List[str]) error!'
        if isinstance(candidate_relations, str):
            candidate_relations = [candidate_relations]
        assert isinstance(candidate_relations, List), f'{error_msg} type of `candidate_relations` must be List, not {type(candidate_relations)}.'
    
    if isinstance(object, Set):
        ret_rel = set()
        for obj in object:
            ret_rel.update(query_relation_agent(obj, candidate_relations))
        return list(ret_rel)
    
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    return query_relation(object, None, candidate_relations, is_anchor_agent=True)

def query_attribute(object: ObjectAttribute, attribute_type: Optional[str]=None, candidate_attribute_values: Optional[List[str]]=None) -> Union[List[float], float, str]:
    """
    Returns the attribute of the object.

    Parameters
    ----------
    object : The object.
    attribute : The attribute of the object. Must be chosen from the following list:
        ["category", "xyz", "lwh", "direction", "caption", "distance"]
    """
    legal_attributes = ["category", "xyz", "direction", "lwh", "distance"]
    show_attributes = ["lwh", "distance"]
    if use_openshape:
        legal_attributes.extend(list(attribute_candidates.keys()))
        show_attributes.extend(["color", "shape", "material"])
        error_msg = 'Calling query_attribute(object: ObjectAttribute, attribute_type: str, candidate_attribute_values: List[str]) error!'
    else:
        error_msg = 'Calling query_attribute(object: ObjectAttribute, attribute_type: str) error!'
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    assert attribute_type or candidate_attribute_values, f'{error_msg} `attribute_type` and `candidate_attribute_values` cannot both be None.'
    if (attribute_type and candidate_attribute_values==None) or (not use_openshape):
        assert attribute_type in legal_attributes, f'{error_msg} `attribute_type` must be chosen from {show_attributes}, not {attribute_type}.'
    
    if attribute_type in ["lwh", "caption", "category", "xyz"]:
        return object.__getattribute__(attribute_type)
    if attribute_type == "distance":
        if object.distance:
            distance = object.distance
        else:
            target_object = scene_data.get_object(object.object_id)
            distance = agent.distance_from_other_object(target_object)
            object.distance = distance
        return distance
    if candidate_attribute_values==None or len(candidate_attribute_values)<=1:
        if candidate_attribute_values and len(candidate_attribute_values)==1:
            if equal_attribute_value(object, candidate_attribute_values[0]):
                return candidate_attribute_values[0]
        if attribute_type in attribute_candidates.keys():
            return query_attribute_v2(object, attribute_type)
    else:
        for candidate_attribute_value in candidate_attribute_values:
            assert isinstance(candidate_attribute_value, str), f'{error_msg} type of each element in `candidate_attribute_values` must be str, not {type(candidate_attribute_value)}.'
        return query_attribute_v2(object, candidate_attribute_values)

def query_caption(object: ObjectAttribute) -> str:
    error_msg = 'Calling query_caption(object: ObjectAttribute) -> str error!'
    assert use_caption, "NameError: name 'query_caption' is not defined"
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    return object.caption    

def query_state(object: ObjectAttribute, candidate_states: List[str]) -> str:
    error_msg = 'query_state(object: ObjectAttribute, candidate_states: List[str]) error!'
    assert use_openshape, "NameError: name 'query_state' is not defined"
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object must` be ObjectAttribute, not {type(object)}.'
    assert isinstance(candidate_states, List), f'{error_msg} type of `candidate_states` must be List, not {type(candidate_states)}.'
    for state in candidate_states:
        assert isinstance(state, str), f'{error_msg} type of each element in `candidate_states` must be str, not {type(state)} (your state: {state}).'
    return query_attribute_v2(object, candidate_states)

def query_color(object: ObjectAttribute, candidate_colors: List[str]=None) -> str:
    error_msg = 'query_color(object: ObjectAttribute, candidate_colors: List[str]) error!'
    assert use_openshape, "NameError: name 'query_color' is not defined"
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    if candidate_colors:
        assert isinstance(candidate_colors, List), f'{error_msg} type of `candidate_colors` must be List, not {type(candidate_colors)}.'
        for color in candidate_colors:
            assert isinstance(color, str), f'{error_msg} type of each element in `candidate_colors` must be str, not {type(color)} (your color: {color}).'
       
    return query_attribute(object, "color", candidate_colors)

def query_material(object: ObjectAttribute, candidate_materials: List[str]=None) -> str:
    error_msg = 'query_material(object: ObjectAttribute, candidate_materials: List[str]) error!'
    assert use_openshape, "NameError: name 'query_material' is not defined"
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    if candidate_materials:
        assert isinstance(candidate_materials, List), f'{error_msg} type of `candidate_materials` must be List, not {type(candidate_materials)}.'
        for material in candidate_materials:
            assert isinstance(material, str), f'{error_msg} type of each element in `candidate_materials` must be str, not {type(material)} (your material: {material}).'
        
    return query_attribute(object, "material", candidate_materials)

def query_shape(object: ObjectAttribute, candidate_shapes: List[str]=None) -> str:
    error_msg = 'query_shape(object: ObjectAttribute, candidate_shapes: List[str]) error!'
    assert use_openshape, "NameError: name 'query_shape' is not defined"
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    if candidate_shapes:
        assert isinstance(candidate_shapes, List), f'{error_msg} type of `candidate_shapes` must be List, not {type(candidate_shapes)}.'
        for shape in candidate_shapes:
            assert isinstance(shape, str), f'{error_msg} type of each element in `candidate_shapes` must be str, not {type(shape)} (your shape: {shape}).'
        return query_attribute_v2(object, candidate_shapes)
    
    return query_attribute(object, "shape", candidate_shapes)

def query_bbox(object: ObjectAttribute) -> List[float]:
    error_msg = 'query_bbox(object: ObjectAttribute) -> List[float] error!'
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    return object.lwh

def query_distance(object: ObjectAttribute) -> float:
    error_msg = 'query_distance(object: ObjectAttribute) -> float error!'
    assert isinstance(object, ObjectAttribute), f'{error_msg} type of `object` must be ObjectAttribute, not {type(object)}.'
    return query_attribute(object, "distance")

def query_attribute_v2(object: ObjectAttribute, attribute: Union[str, List[str]]) -> str:
    object_3d = scene_data.get_object(object.object_id)
    if isinstance(attribute, str) and attribute in list(attribute_candidates.keys()):
        attr = match_sqa3d_attr.classify(object_3d.point_clouds, object_3d.rgbs, attribute_candidates[attribute])
        return attr
    elif isinstance(attribute, List):
        return match_sqa3d_attr.classify(object_3d.point_clouds, object_3d.rgbs, attribute)

def equal_attribute_value(object: ObjectAttribute, attribute_value: str) -> bool:
    object_3d = scene_data.get_object(object.object_id)
    return match_sqa3d_attr.match(object_3d.point_clouds, object_3d.rgbs, attribute_value)


if __name__=="__main__":
    scene_dir = "../data/scans"
    scene_id = "scene0000_00"
    question_id = "220602000226" 
    question = "The bathroom is on which side of me?"
    situation = "I am sitting on a couch facing the coffee table while there is a bicycle and a curtain on my left."
    answers = ["right"]
    base_scene_dir = os.path.join(scene_dir, scene_id)
    mesh_file = os.path.join(base_scene_dir, scene_id+"_vh_clean_2.ply")
    agg_file = os.path.join(base_scene_dir, scene_id+".aggregation.json")
    seg_file = os.path.join(base_scene_dir, scene_id+"_vh_clean_2.0.010000.segs.json")
    label_map_file = "../data/scannetv2-labels.combined.tsv"
    position = [-1.1113222951356774, 0.7810304200608889, 0, 0, 0, 0.6118578909427186, -0.790967711914415]
    object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs = export(mesh_file, agg_file, seg_file, label_map_file, position)
    scene_data = Scene(object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, question, answers)
    
    anchor_object_a = scene_data.get_object(4)
    anchor_object_b = scene_data.get_object(3)
    target_object = scene_data.get_object(8)
    target_object_list = [scene_data.get_object(i) for i in range(7, 15)]


    HpHelper = HorizontalProximityHelper()
    VpHelper = VerticalProximityHelper()
    BtHelper = BetweenHelper()
    AlHelper = AllocentricHelper()

    ret_hp = HpHelper.infer(target_object_list, anchor_object_a)

    for i in range(len(scene_data.get_object_num())):
        if i==7:
            continue
        anchor_object_a = scene_data.get_object(i)
        ret_vp = VpHelper.infer(target_object, anchor_object_a)
        ret_bt = BtHelper.infer(target_object, anchor_object_a, anchor_object_b)
        ret_al = AlHelper.infer(target_object, anchor_object_a)

    match_sqa3d_attr = Match_SQA3D_Attr()
    