import sys
from tqdm import tqdm
import json
import os
import argparse
import sys
sys.path.append("../src")
from LLMTPC_agent.LLMTPC_SOP import LLMTPC_SOP as SOP
from LLMTPC_agent.agent import LLMTPC_Agent as Agent
from LLMTPC_agent.environment import LLMTPC_Environment as Environment
from dataset.scene_dataset import SceneDataset
from LLMTPC_executor.api.api import Match_SQA3D_Attr


def init(config):     
    sop = SOP.from_config(config)
    agents,roles_to_names,names_to_roles = Agent.from_config(config)
    environment = Environment.from_config(config)
    environment.agents = agents
    environment.roles_to_names,environment.names_to_roles = roles_to_names,names_to_roles
    sop.roles_to_names,sop.names_to_roles = roles_to_names,names_to_roles
    for name,agent in agents.items():
        agent.environment = environment
    return agents,sop,environment

def init_scene_dataset(config):
    scene_dataset = SceneDataset.from_config(config["config"]["scene_info"])
    return scene_dataset

def init_openshape_matcher(config):
    openshape_matcher = Match_SQA3D_Attr(
        config_path=config["config"]["openshape_config"]["meta_info"]["openshape_config_path"],
        openclip_model_path=config["config"]["openshape_config"]["meta_info"]["openclip_model_path"],
        openshape_model_path=config["config"]["openshape_config"]["meta_info"]["openshape_model_path"]
    )
    openshape_matcher.init_model()
    return openshape_matcher
    
def run(agents: Agent, sop: SOP, environment: Environment):
    action = None
    while True:
        current_state,current_agent= sop.next(environment, agents, action)   # State, Agent
        if sop.finished:
            # print("finished!")
            os.environ.clear()
            break
        user_input = input(f"{current_agent.name}:") if current_agent.is_user else ""
        action = current_agent.step(current_state,user_input, action)   #component_dict = current_state[self.role[current_node.name]]   current_agent.complete(component_dict) 
        memory = action.process()
        environment.update_memory(memory, current_state, current_agent)
        
parser = argparse.ArgumentParser(description='LLM-TPC')
parser.add_argument('--agent', type=str, help='path to SOP json')
args = parser.parse_args()

with open(args.agent, 'r') as f:
    config = json.load(f)
scene_dataset = init_scene_dataset(config)
config["config"]["scene_info"]["scene_dataset"] = scene_dataset
if config["config"]["setting"]["use_openshape"]:
    config["config"]["openshape_config"]["attr_matcher"] = init_openshape_matcher(config)


scene_id = config["config"]["scene_info"]["qa_info"]["scene_id"]
question_id = config["config"]["scene_info"]["qa_info"]["question_id"]
scene = scene_dataset.get_scene(scene_id, question_id)
config["config"]["scene_info"]["scene"] = scene
agents,sop,environment = init(config)
run(agents,sop,environment)
print("finished!")