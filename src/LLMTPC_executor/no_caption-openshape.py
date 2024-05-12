import re
import sys
from io import StringIO
import traceback
import json
from api.api import *
import api.api
from tqdm import tqdm


scene_dir = "../../data/scans"
label_map_file = "../../data/scannetv2-labels.combined.tsv"
caption_file = "../../data/ScanRefer_organized.json"
qa_file = "../../data/qa/SQA_train.json"

openshape_config_path = "../../src/LLMTPC_executor/api/openshape/configs/train.yaml"
openshape_model_path  = "../../data/openshape/model.pt"
openclip_model_path = "../../data/openshape/open_clip_pytorch_model.bin"

txt_file = "no_caption-openshape-few_shot_example.txt"
write_out = open(txt_file, "w")

def prepare_qa(qa_file):
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

def prepare_caption(caption_file):
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

def init_openshape_matcher(openshape_config_path, openclip_model_path, openshape_model_path):
    openshape_matcher = Match_SQA3D_Attr(
        config_path=openshape_config_path,
        openclip_model_path=openclip_model_path,
        openshape_model_path=openshape_model_path
    )
    openshape_matcher.init_model()
    return openshape_matcher

qa = prepare_qa(qa_file)
all_caption = prepare_caption(caption_file)
api.api.match_sqa3d_attr = init_openshape_matcher(openshape_config_path, openclip_model_path, openshape_model_path)

def prepare_scene_data(scene_id, question_id):
    base_scene_dir = os.path.join(scene_dir, scene_id)
    mesh_file = os.path.join(base_scene_dir, scene_id+"_vh_clean_2.ply")
    agg_file = os.path.join(base_scene_dir, scene_id+".aggregation.json")
    seg_file = os.path.join(base_scene_dir, scene_id+"_vh_clean_2.0.010000.segs.json")
    position = qa[scene_id][question_id]["position"]
    situation = qa[scene_id][question_id]["situation"]
    answers = qa[scene_id][question_id]["answers"]
    question = qa[scene_id][question_id]["question"]
    object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs = export(mesh_file, agg_file, seg_file, label_map_file, position)
    return object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, all_caption[scene_id], question, answers

def generate_query(scene_data: Scene):
    objects = []
    for obj, num in scene_data.get_object_num().items():
        objects.append(str(num)+" "+obj)
    answer = scene_data.answers[0]
    template = f"I am in a room. Looking around me, I see some objects: {', '.join(objects)}.\nMy situation: {scene_data.situation}\nQuestion: {scene_data.question}"
    return template, answer

def extract(string):
    match_thought = re.search(r"Thought:\s*(.*?)\nAction:", string, re.DOTALL)
    thought = match_thought.group(1)

    match_action = re.search(r"Action:\s+(.*?)\nAction Input:", string)
    action = match_action.group(1)

    if action == "Program":
        match_action_input = re.search(r"Action Input:\s*```Python\n(.*?)\n```", string, re.DOTALL)
    else:
        match_action_input = re.search(r"Action Input:\s*(.*?)$", string)
    action_input = match_action_input.group(1)
    return {"thought": thought, "action": action, "action_input": action_input}

def execute_program(program, global_vars=globals()):
    exec_result = {"execution state": "SUCCESS", "message": ""}
    old_stdout = sys.stdout
    new_stdout = StringIO()
    sys.stdout = new_stdout    
    try:
        exec(program, global_vars)
    except Exception as e:
        error_message = traceback.format_exc().strip().split("\n")[-1]
        print(error_message)
        exec_result["execution state"] = "ERROR"

    output = new_stdout.getvalue()
    exec_result["message"] = output.strip()
    sys.stdout = old_stdout
    return exec_result

def print_query(example):
    query, answer = generate_query(api.api.scene_data)
    print("-"*20, "query", "-"*20)
    print(query)
    print(answer)
    print("-"*20, "query", "-"*20, file=write_out)
    print(query, file=write_out)
    print(answer, file=write_out)
    example.append({"role": "system", "name": "example_user", "content": query})

def print_thought(thought, example, end=False):
    print("-"*20, "thought", "-"*20)
    print(thought)
    print("-"*20, "thought", "-"*20, file=write_out)
    print(thought, file=write_out)
    example.append({"role": "system", "name": "example_assistant", "content": thought})
    if end:
        print("\n\n\n")
        print("\n\n\n", file=write_out)

def print_observation(thought, example):
    print("-"*20, "observation", "-"*20)
    print("-"*20, "observation", "-"*20, file=write_out)
    res_dict = extract(thought)
    program = res_dict["action_input"]
    exec_res = execute_program(program)
    observation = f"Observation: {exec_res['message']}"
    print(observation)
    print(observation, file=write_out)
    example.append({"role": "system", "name": "example_user", "content": observation})
    

few_shot_example = []


##############################################################################################################
##############################################################################################################
##############################################################################################################
scene_id = "scene0536_00"
question_id = "220602000413"
# question = "What is behind me directly?"
# answers = ["coffee table"]
object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, caption, question, answers = prepare_scene_data(scene_id, question_id)
api.api.scene_data = Scene(object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, question, answers, caption)
example = []

print_query(example)

program = """\
# Get object set in the scene
object_set = scene()

# Identify objects behind me
object_behind_set = relate_agent(object_set=object_set, relation="behind")

# Sort the objects behind me by distance
object_behind_by_distance = list(object_behind_set)
object_behind_by_distance.sort(key=lambda x: query_attribute(x, "distance"))

# Determine what object is directly behind me
category_behind_by_distance = [obj.category for obj in object_behind_by_distance][:3]
print(f"Objects directly behind me: {category_behind_by_distance}")
"""
# Question: What is behind me directly?
thought = f"""\
Thought: \
Based on the information provided, I need to find the object behind me and close to me. However, I'm not sure what object is directly behind me. I will develop a program to determine what is directly behind me using the following steps:
Step 1. Identify objects behind me. (Program)
Step 2. Sort the objects behind me by distance. (Program)
Step 3. Determine what object is directly behind me. (Wait for the execution result before proceeding with further analysis)
Action: Program
Action Input:
```Python
{program}\
```\
"""
print_thought(thought, example)
print_observation(thought, example)
thought = f"""\
Thought: Based on the observation, there are multiple objects behind me in proximity: coffee table, couch, pillow. \
Considering my situation, I am facing a chair, while having the couch on my right and the coffee table behind me. Therefore, the coffee table is located directly behind me.
I got the final answer to the question: "What is behind me directly?"
Action: Final Answer
Action Input: coffee table\
"""
print_thought(thought, example, True)
few_shot_example.append(example)


##############################################################################################################
##############################################################################################################
##############################################################################################################
scene_id = "scene0503_00"
question_id = "220602028843"
# question = "Which direction should I go if I want to exit the room?"
# answers = ["left"]
object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, caption, question, answers = prepare_scene_data(scene_id, question_id)
api.api.scene_data = Scene(object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, question, answers, caption)
example = []

print_query(example)

program = """\
# Get object set in the scene
object_set = scene()

# Filter all the doors
door_set = filter(object_set=object_set, category="door")

for door in door_set:
    # Determine the direction of the door relative to my current position
    door_direction = query_relation_agent(object=door, candidate_relations=["left", "right", "forward", "backward"])
    print(f"Direction of the door relative to my current position: {door_direction}")

    # Decide which direction I should go to exit the room
    print(f"I should go {' '.join(door_direction)} to exit the room")
"""
# Question: Which direction should I go if I want to exit the room?
thought = f"""\
Thought: To exit the room, I should find the door. I will develop a program to determine the direction of the door using the following steps:
Step 1. Filter all the doors. (Program)
Step 2. Determine the direction of the door relative to my current position. (Program)
Step 3. Decide which direction I should go to exit the room. (Wait for the execution result before proceeding with further analysis)
Action: Program
Action Input:
```Python
{program}\
```\
"""
print_thought(thought, example)
print_observation(thought, example)
thought = """\
Thought: Based on the observation, there is a door on my left back. I should go left, backward to exit the room.
I got the final answer to the question: "Which direction should I go if I want to exit the room?"
Action: Final Answer
Action Input: left backward\
"""
print_thought(thought, example, True)
few_shot_example.append(example)


##############################################################################################################
##############################################################################################################
##############################################################################################################
scene_id = "scene0513_00"
question_id = "220602019345"
# question = "Is the shape of tables on my right round, square or rectangular?"
# answers = ["round"]
object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, caption, question, answers = prepare_scene_data(scene_id, question_id)
api.api.scene_data = Scene(object_pcds, object_labels, object_ids, instance_bboxes, instance_rgbs, situation, question, answers, caption)
example = []

print_query(example)

program = """\
# Get object set in the scene
object_set = scene()

# Filter the tables from the objects in the scene
table_set = filter(object_set=object_set, category="table")

# Find the tables on my right
table_on_my_right_set = relate_agent(object_set=table_set, relation="right")

# Determine if the shape of tables on my right are round, square or rectangular
shape_table_on_my_right = []
for table in table_on_my_right_set:
    shape = query_attribute(object=table, attribute_type="shape", candidate_attribute_values=["round", "square", "rectangular"])
    shape_table_on_my_right.append(shape)
print(f"Shape of tables on my right: {shape_table_on_my_right}")
"""
# Question: Is the shape of tables on my right round, square or rectangular?
thought = f"""\
Thought: To determine whether the shape of tables on my right is round, square or rectangular, \
I need to find the tables on my right, and then query the shape of each table to determine whether they are round, square or rectangular. \
I will develop a program to accomplish this using the following steps:
Step 1. Filter the tables from the objects in the scene. (Program)
Step 2. Find the tables on my right. (Program)
Step 3. Determine if the shape of tables on my right are round, square or rectangular. (Wait for the execution result before proceeding with further analysis)
Action: Program
Action Input:
```Python
{program}\
```\
"""
print_thought(thought, example)
print_observation(thought, example)
thought = f"""\
Thought: Based on the observation, the shape of the tables on my right is round.
I got the final answer to the question: "Is the shape of tables on my right round, square or rectangular?"
Action: Final Answer
Action Input: round\
"""
print_thought(thought, example, True)
few_shot_example.append(example)


base_file = "few_shot_example.json"
with open(base_file, 'w') as f:
    json.dump(few_shot_example, f, indent=4)

base_dir = "../../scripts/no_caption-openshape-gt_seg-gt_label"
with open(os.path.join(base_dir, base_file), 'w') as f:
    json.dump(few_shot_example, f, indent=4)

write_out.close()

