import sys
sys.path.append("../src/LLMTPC_agent")
import os
from memory import LLMTPC_Memory as Memory
from LLM import OpenAILLM
from utils import *

class LLMTPC_Environment:
    """
    The place where the agent activities, responsible for storing some shared memories
    """
    def __init__(self, config) -> None:
        self.shared_memory = {"long_term_memory": [], "short_term_memory": None}
        self.agents = None

        self.summary_system_prompt = {}
        self.summary_last_prompt = {}
        self.environment_prompt = {}
        self.environment_type = config["environment_type"] if "environment_type" in config else "cooperative"
        self.current_chat_history_idx = 0
        self.LLMs = {}

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        
        if "scene_info" in config["config"]:
            scene_info = config["config"]['scene_info']
            scene_id = scene_info["qa_info"]["scene_id"]
            question_id = scene_info["qa_info"]["question_id"]
            file_name = f"{scene_id}_{question_id}-{timestamp}.json"
        else:
            file_name = f"{timestamp}.json"
        self.log_file = f"{config['log_path']}/{file_name}" if "log_path" in config else f"logs/log/{file_name}"
        
        # Initialize the summary method for each state
        for state_name, state_dict in config["states"].items():
            if state_name != "end_state":
                self.summary_system_prompt[state_name] = (
                    state_dict["summary_system_prompt"]
                    if "summary_system_prompt" in state_dict
                    else " "
                )

                self.summary_last_prompt[state_name] = (
                    state_dict["summary_last_prompt"]
                    if "summary_last_prompt" in state_dict
                    else " "
                )

                self.environment_prompt[state_name] = (
                    state_dict["environment_prompt"]
                    if "environment_prompt" in state_dict
                    else " "
                )
                LLM_type = (
                    state_dict["LLM_type"] if "LLM_type" in state_dict else "OpenAI"
                )
                if LLM_type == "OpenAI":
                    if "LLM" in state_dict:
                        self.LLMs[state_name] = OpenAILLM(**state_dict["LLM"])
                    else:
                        self.LLMs[state_name] = OpenAILLM(model = "gpt-3.5-turbo-16k-0613",temperature=0.3,log_path=f"logs/{state_name}")
        self.roles_to_names = None
        self.names_to_roles = None

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def update_memory(self, memory, current_state, current_agent):
        """
        update chat embbedings and long term memory,short term memory,agents long term memory
        """
        MAX_CHAT_HISTORY = eval(os.environ["MAX_CHAT_HISTORY"])
        self.shared_memory["long_term_memory"].extend(memory)
        self.agents[current_agent.name].update_memory(memory)
        save_logs(self.log_file, memory)
    
    
    def _get_agent_last_conversation_idx(self,agent,current_long_term_memory):
        last_conversation_idx = -1
        for i, history in enumerate(current_long_term_memory):
            if history.send_name == agent.name:
                last_conversation_idx = i
        return last_conversation_idx
