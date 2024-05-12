import sys
sys.path.append("../src/LLMTPC_agent")
import json
import os
from action import LLMTPC_Action
from LLM import OpenAILLM
from component import *
import random

class LLMTPC_Agent:
    def __init__(self, name, agent_state_roles, **kwargs) -> None:
        self.state_roles = agent_state_roles
        self.name = name
        
        self.style = kwargs["style"]
        self.LLMs = kwargs["LLMs"]
        self.LLM = None
        self.is_user = kwargs["is_user"]
        self.begins = kwargs["begins"] if "begins" in kwargs else False
        self.current_role = ""
        self.long_term_memory = []
        self.short_term_memory = ""
        self.current_state = None
        self.first_speak = True
        self.environment = None
        self.scene_info = kwargs["scene_info"] if "scene_info" in kwargs else None
        self.openshape_config = kwargs["openshape_config"] if "openshape_config" in kwargs else None
        self.setting = kwargs["setting"] if "setting" in kwargs else None
    

    @classmethod
    def from_config(cls, config):
        """
        Initialize agents based on json file
        Return:
        agents(dict) : key:agent_name;value:class(Agent) 
        names_to_roles(dict) : key:state_name  value:(dict; (key:agent_name ; value:agent_role))
        roles_to_names(dict) : key:state_name  value:(dict; (key:agent_role ; value:agent_name))
        """        
        roles_to_names = {}
        names_to_roles = {}
        agents = {}
        user_names = json.loads(os.environ["User_Names"]) if "User_Names" in os.environ else []
        for agent_name, agent_dict in config["agents"].items():
            agent_state_roles = {}
            agent_LLMs = {}
            agent_begins = {}
            for state_name, agent_role in agent_dict["roles"].items():
                
                agent_begins[state_name] = {}
                
                if state_name not in roles_to_names:
                    roles_to_names[state_name] = {}
                if state_name not in names_to_roles:
                    names_to_roles[state_name] = {}
                roles_to_names[state_name][agent_role] = agent_name
                names_to_roles[state_name][agent_name] = agent_role
                agent_state_roles[state_name] = agent_role
                current_state = config["states"][state_name]
                
                current_state_begin_role = current_state["begin_role"] if "begin_role" in current_state else current_state["roles"][0]
                agent_begins[state_name]["is_begin"] = current_state_begin_role==agent_role if ("begin_role" in current_state) and "begin_query" in current_state else False
                agent_begins[state_name]["begin_query"] = current_state["begin_query"] if "begin_query" in current_state else ""
                
                agent_LLMs[state_name] = OpenAILLM(**config["LLM"])
               
            agents[agent_name] = cls(
                agent_name,
                agent_state_roles,
                LLMs=agent_LLMs,
                is_user=agent_name in user_names,
                style = agent_dict["style"],
                begins = agent_begins,
                scene_info = config["config"]["scene_info"],
                openshape_config = config["config"]["openshape_config"],
                setting = config["config"]["setting"]
            )
        assert len(config["agents"].keys()) != 2 or (roles_to_names[config["root"]][config["states"][config["root"]]["begin_role"]] not in user_names and "begin_query"  in config["states"][config["root"]]),"In a single-agent scenario, there must be an opening statement and it must be the agent" 
        return agents, roles_to_names, names_to_roles

    def step(self, current_state, input="", last_action_res=None):
        """
        return actions by current state and environment
        Return: action(Action)
        """
        current_state.chat_nums +=1
        state_begin = current_state.is_begin
        agent_begin = self.begins[current_state.name]["is_begin"]
        self.begins[current_state.name]["is_begin"] = False
        current_state.is_begin = False
        environment = self.environment
        
        self.current_state = current_state
        # First update the information according to the current environment
        
        response = []
        res_dict = {}
        
        if agent_begin:
            response = []


        if self.is_user:
            response = f"{self.name}:{input}"
        else:
            if agent_begin:
                response, res_dict = self.preprocess(last_action_res)
            else:
                response, res_dict = self.act(last_action_res)
        res_dict["scene_info"] = self.scene_info
        res_dict["attr_matcher"] = self.openshape_config["attr_matcher"]
        res_dict["setting"] = self.setting

        action_dict =  {
            "response": response,
            "res_dict": res_dict,
            "role": self.state_roles[current_state.name],
            "name": self.name,
            "state_begin" : state_begin,
            "agent_begin" : agent_begin,
            "is_user" : self.is_user,
            "state": current_state
        }
        return  LLMTPC_Action(**action_dict)

    def act(self, last_action_res):
        """
        return actions by the current state
        """
        current_state = self.current_state
        chat_history = self.long_term_memory
        current_LLM = self.LLMs[current_state.name]
        
        res_dict = self.compile(last_action_res)
        response = None
        if res_dict["need_response"]:
            response = current_LLM.get_response(
                chat_history, None, None, stream=True, res_dict=res_dict, state=current_state.name, debug=random.choice([True, False])
            )
        return response,res_dict
    
    def preprocess(self, last_action_res):
        """
        return actions by the current state
        """        
        current_state = self.current_state
        self.current_roles = self.state_roles[current_state.name]
        current_state_name = current_state.name
        components = current_state.components[self.state_roles[current_state_name]]

        response = None
        res_dict = {}
        
        if current_state.name == "start_state":
            response, res_dict = self._preprocess_start_state(components, last_action_res)
        elif current_state.name == "query_state":
            response, res_dict = self._preprocess_query_state(components, last_action_res)
        elif current_state.name == "postprocessing_state":
            response, res_dict = self._preprocess_postprocessing_state(components, last_action_res)
        else:
            response, res_dict = self._preprocess(components, last_action_res)
        return response,res_dict

    def update_memory(self, memory):
        for mem in memory:
            self.long_term_memory.append(
                {"role": mem.system_role, "content": mem.content}
            )     
        
    def compile(self, last_action_res):
        """
        get prompt from state depend on your role
        Return:
        system_prompt:system_prompt for agents's LLM
        last_prompt:last_prompt for agents's LLM
        res_dict(dict): Other return from tool component.
        """
        current_state = self.current_state
        self.current_roles = self.state_roles[current_state.name]
        current_state_name = current_state.name
        self.LLM = self.LLMs[current_state_name]
        components = current_state.components[self.state_roles[current_state_name]]

        res_dict = {}
        if current_state.name == "TPC_state":
            res_dict = self._compile_TPC_state(components, last_action_res)
        else:
            res_dict = self._compile(components, last_action_res)

        return res_dict
    
    def _preprocess(self, components, last_action_res):
        response = None
        res_dict = {}
        current_state = self.current_state
        begin_query = self.begins[current_state.name]["begin_query"]
        response = begin_query
        return response, res_dict
    
    def _preprocess_start_state(self, components, last_action_res):
        response = None
        res_dict = {}
        current_state = self.current_state
        begin_query = self.begins[current_state.name]["begin_query"]
        new_chat_history = []
        new_chat_history.append({"role": "system", "content": begin_query})
        if "few_shot_example" in components:
            new_chat_history.extend(components["few_shot_example"].get_few_shot_example())
        res_dict["new_chat_history"] = new_chat_history
        return response, res_dict
    
    def _preprocess_query_state(self, components, last_action_res):
        response = None
        res_dict = {}
        current_state = self.current_state
        begin_query = self.begins[current_state.name]["begin_query"]
        new_chat_history = []
        new_chat_history.append({"role": "system", "content": begin_query})
        if "query" in components:
            new_chat_history.append(components["query"].get_message())
        res_dict["new_chat_history"] = new_chat_history
        return response, res_dict

    def _preprocess_postprocessing_state(self, components, last_action_res):
        response = None
        res_dict = {}
        current_state = self.current_state
        begin_query = self.begins[current_state.name]["begin_query"]

        if "Action" in last_action_res.parse_result and last_action_res.parse_result["Action"] == "Final Answer":
            res_dict["Final Answer"] = last_action_res.parse_result["Action Input"]
        else:
            new_chat_history = []
            content = begin_query
            new_chat_history.append({"role": "user", "content": content})
            res_dict["new_chat_history"] = new_chat_history
        return response, res_dict

    def _compile_TPC_state(self, components, last_action_res):
        res_dict = {
            "need_response": True,
            "debug_message": components["debug_message"],
            "parse_message": components["parse_message"]
        }
        return res_dict

    def _compile(self, components, last_action_res):
        """
        get prompt from state depend on your role
        Return:
        system_prompt: system_prompt for agents's LLM
        last_prompt: last_prompt for agents's LLM
        res_dict(dict): Other return from tool component.
        """
    
        new_chat_history = []
        res_dict = {
            "need_response": True
        }
        
        if "start_message" in components:
            component = components["start_message"]
            new_chat_history.append(component.get_message())
        if "last_message" in components:
            component = components["last_message"]
            new_chat_history.append(component.get_message())

        res_dict["new_chat_history"] = new_chat_history
        return res_dict

    def observe(self):
        """
        Update one's own memory according to the current environment, including: updating short-term memory; updating long-term memory
        """
        return self.environment._observe(self)