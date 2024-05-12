import sys
sys.path.append("../src/LLMTPC_agent")
import random
import os
from utils import save_logs
from state import LLMTPC_State as State
from action import parse_response
from LLM import OpenAILLM

class LLMTPC_SOP:
    """
    Responsible for managing the operational processes of all agents
    """

    # SOP should have args : "states" "relations" "root"

    def __init__(self, **kwargs):
        self.controller_dict = {}
        LLM_type = kwargs["LLM_type"] if "LLM_type" in kwargs else "OpenAI"
        if LLM_type == "OpenAI":
            self.LLM = (
                OpenAILLM(**kwargs["LLM"])
                if "LLM" in kwargs
                else OpenAILLM(model = "gpt-3.5-turbo-16k-0613",temperature=0.3,log_path="logs/god")
            )

        self.states = {}
        self.init_states(kwargs["states"], kwargs["config"])
        self.init_relation(kwargs["relations"])
        for state_name, states_dict in kwargs["states"].items():
            if state_name != "end_state" and "controller" in states_dict:
                self.controller_dict[state_name] = states_dict["controller"]

        self.user_names = kwargs["user_names"] if "user_names" in kwargs else []
        self.root = self.states[kwargs["root"]]
        self.current_state = self.root
        self.finish_state_name = (
            kwargs["finish_state_name"]
            if "finish_state_name" in kwargs
            else "end_state"
        )
        self.final_state_name = (
            kwargs["final_state_name"]
            if "final_state_name" in kwargs
            else "end_state"
        )
        self.roles_to_names = None
        self.names_to_roles = None
        self.finished = False
        self.scene_info = kwargs["config"]["scene_info"]
        self.setting = kwargs["config"]["setting"]

    @classmethod
    def from_config(cls, config):
        for key,value in config["config"].items():
            if isinstance(value, str):
                os.environ[key] = value
        assert "API_KEY" in os.environ and os.environ["API_KEY"] != "API_KEY","Please go to config.json to set API_KEY"
        assert "PROXY" in os.environ and os.environ["PROXY"] != "PROXY","Please go to config.json to set PROXY"
        
        sop = LLMTPC_SOP(**config)
        return sop

    def init_states(self, states_dict, config_dict):
        for state_name, state_dict in states_dict.items():
            state_dict["name"] = state_name
            state_dict["scene_info"] = config_dict["scene_info"]
            self.states[state_name] = State(**state_dict)

    def init_relation(self, relations):
        for state_name, state_relation in relations.items():
            for idx, next_state_name in state_relation.items():
                self.states[state_name].next_states[idx] = self.states[next_state_name]

    def transit(self, chat_history, **kwargs):
        """
        Determine the next state based on the current situation
        Return : 
        next_state(State) : the next state
        """
        action = kwargs["action"]
        current_state = self.current_state
        controller_dict = self.controller_dict[current_state.name]
        max_chat_nums = controller_dict["max_chat_nums"] if "max_chat_nums" in controller_dict else 1000

        # If it is a single loop node, just keep looping
        if len(self.current_state.next_states) == 1:
            next_state = "0"
        elif len(chat_history)>=int(os.environ["MAX_CHAT_HISTORY"]):
            if current_state.name!=self.final_state_name:   # states before postprocessing_state
                next_state = self.states[self.final_state_name] # next_state = postprocessing_state
            else:   # current_state is postprocessing_state
                if current_state.chat_nums>=max_chat_nums:  # Having tried the last time to get the final answer. Go to the end_state.
                    next_state = self.states[self.finish_state_name]
                else:   # If the final answer is not obtained, then try once more to get it. Otherwise, go to the end_state.
                    next_state = self.current_state.next_states[self._transit_from_postprocessing_state(chat_history, action)]
        # Otherwise, the controller needs to determine which node to enter.   
        else: 
            if current_state.chat_nums>=max_chat_nums:
                next_state = "1"
            else:
                next_state = "0"
                if current_state.name == "TPC_state":
                    next_state = self._transit_from_TPC_state(chat_history, action)
             
                elif current_state.name == "postprocessing_state":
                    next_state = self._transit_from_postprocessing_state(chat_history, action)
                        
            next_state = self.current_state.next_states[next_state]
        return next_state

    def _transit_from_TPC_state(self, chat_history, action):
        next_state = "0"    # TPC_state
        parse_result = action.parse_result

        if parse_result["Action"] == "Final Answer":
            next_state = "1"    # postprocessing_state
        return next_state


    def _transit_from_postprocessing_state(self, chat_history, action):
        next_state = "0"    # postprocessing_state
        if "Final Answer" in action.res_dict:   # Go to the end_state.
            next_state = "1"    # end_state
        # Try once more to get the final answer.
        return next_state

    def route(self, chat_history, **kwargs):
        """
        Determine the role that needs action based on the current situation
        Return : 
        current_agent(Agent) : the next act agent
        """
        agents = kwargs["agents"]
        
        # Start assigning roles after knowing which state you have entered. If there is only one role in that state, assign it directly to him.
        if len(self.current_state.roles) == 1:
            next_role = self.current_state.roles[0]
        
        # Otherwise the controller determines
        else:
            relevant_history = kwargs["relevant_history"]
            controller_type = (
                self.controller_dict[self.current_state.name]["controller_type"]
                if "controller_type" in self.controller_dict[self.current_state.name]
                else "order"
            )
            
            # Speak in order
            if controller_type == "order":
                # If there is no begin role, it will be given directly to the first person.
                if not self.current_state.current_role:
                    next_role = self.current_state.roles[0]
                # otherwise first
                else:
                    self.current_state.index += 1
                    self.current_state.index =  (self.current_state.index) % len(self.current_state.roles)
                    next_role = self.current_state.roles[self.current_state.index]
            # random speak
            elif controller_type == "random":
                next_role = random.choice(self.current_state.roles)
            
        # If the next character is not available, pick one at random    
        if next_role not in self.current_state.roles:
            next_role = random.choice(self.current_state.roles)
            
        self.current_state.current_role = next_role 
        
        next_agent = agents[self.roles_to_names[self.current_state.name][next_role]]
        
        return next_agent
    
    def next(self, environment, agents, action):
        """
        Determine the next state and the agent that needs action based on the current situation
        """
        # If it is the first time to enter this state
        if self.current_state.is_begin:
            agent_name = self.roles_to_names[self.current_state.name][self.current_state.begin_role]
            agent = agents[agent_name]
            return self.current_state,agent # State, Agent
    
    
        # get relevant history
        relevant_history = environment.shared_memory["long_term_memory"]
        
        next_state = self.transit(
            chat_history=environment.shared_memory["long_term_memory"][
                environment.current_chat_history_idx :
            ],
            # relevant_history=relevant_history,
            environment=environment,
            action=action
        )

        # If you enter the termination node, terminate directly
        if next_state.name == self.finish_state_name:
            if self.states[self.final_state_name].chat_nums>=self.controller_dict[self.states[self.final_state_name].name]["max_chat_nums"]:
                parse_res = parse_response(action.response)
                final_answer = None
                if parse_res["Action"] == "Final Answer":
                    final_answer = parse_res["Action Input"]
                else:
                    final_answer = "Cannot Parse!"
            else:
                final_answer = action.res_dict["Final Answer"]
            
            memory = {
                "send_role": "Final Answer",
                "scene_id": self.scene_info["qa_info"]["scene_id"],
                "question_id": self.scene_info["qa_info"]["question_id"],
                "question": self.scene_info["scene"].question,
                "gt_answer": self.scene_info["scene"].answers[0],
                "answer": final_answer
            }
            save_logs(environment.log_file, [memory])
            self.finished = True
            return None, None

        self.current_state = next_state
        
        # If it is the first time to enter the state and there is a begin query, it will be directly assigned to the begin role.
        if self.current_state.is_begin and self.current_state.begin_role:
            agent_name = self.roles_to_names[self.current_state.name][self.current_state.begin_role]
            agent = agents[agent_name]
            return self.current_state,agent
           

        next_agent = self.route(
            chat_history=environment.shared_memory["long_term_memory"][
                environment.current_chat_history_idx :
            ],
            agents = agents,
            relevant_history=relevant_history,
        )

        return self.current_state, next_agent
