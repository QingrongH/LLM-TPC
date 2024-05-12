import sys
import re
sys.path.append("../src/LLMTPC_agent")
sys.path.append("../src/LLMTPC_executor")
from memory import LLMTPC_Memory
from LLMTPC_executor.executor import execute_program

class LLMTPC_Action:
    """
    The basic action unit of agent
    """
    def __init__(self,**kwargs):
        self.response = None
        self.is_user = False
        self.res_dict = {}
        self.name = ""
        self.role = ""
        for key,value in kwargs.items():
            setattr(self,key,value)
        self.exec_result = {}
        self.parse_result = {}
            
    def process(self):
        """
        processing action
        Rerutn : memory(List(Memory))
        """
        memory = []
        if self.state.name=="start_state":
            memory = self._process_start_state()
        elif self.state.name=="query_state":
            memory = self._process_query_state()
        elif self.state.name=="TPC_state":
            memory = self._process_TPC_state()
        else:
            memory = self._process()
        return memory

    
    def _process_start_state(self):
        memory = []
        send_name = self.name
        send_role = self.role
        if "new_chat_history" in self.res_dict:
            for mem in self.res_dict["new_chat_history"]:
                system_name = mem["name"] if "name" in mem else ""
                memory.append(LLMTPC_Memory(send_role, send_name, mem["content"], mem["role"], system_name))                    
        return memory
    
    def _process_query_state(self):
        memory = []
        send_name = self.name
        send_role = self.role
        if "new_chat_history" in self.res_dict:
            for mem in self.res_dict["new_chat_history"]:
                memory.append(LLMTPC_Memory(send_role, send_name, mem["content"], mem["role"]))                    
        return memory

    def _process_TPC_state(self):        
        memory = []
        response = self.response
        send_name = self.name
        send_role = self.role
        if "new_chat_history" in self.res_dict:
            for mem in self.res_dict["new_chat_history"]:
                memory.append(LLMTPC_Memory(send_role, send_name, mem["content"], mem["role"]))

        memory.append(LLMTPC_Memory(send_role, send_name, response, system_role="assistant"))
        setting = self.res_dict["setting"]
        parse_res = parse_response(response)
        self.parse_result = parse_res
        if parse_res["Action"] == "Program" and parse_res["Action Input"]:  # parse successfully
            scene_info = self.res_dict["scene_info"]
            attr_matcher = self.res_dict["attr_matcher"]
            self._process_TPC_state_program(parse_res, scene_info, attr_matcher, setting, memory)
        elif parse_res["Action"] == "Final Answer":   # final answer
            self._process_TPC_state_final_answer(parse_res, memory)
        else:   # parse error
            self._process_TPC_state_parse_error(parse_res, setting, memory)
        return memory
    

    def _process_TPC_state_program(self, parse_res, scene_info, attr_matcher, setting, memory):
        send_name = self.name
        send_role = self.role
        program = parse_res["Action Input"]
        exec_result = execute_program(program, scene_info, attr_matcher, setting)
        self.exec_result = exec_result

        use_reflection = setting["use_reflection"] if "use_reflection" in setting else True

        if use_reflection:
            self._process_TPC_state_program_with_reflection(send_role, send_name, memory)
        else:
            self._process_TPC_state_program_no_reflection(send_role, send_name, memory)
        
    def _process_TPC_state_program_with_reflection(self, send_role, send_name, memory):
        if self.exec_result["execution state"] == "SUCCESS":    # program executed successfully
            all = "Observation: " + self.exec_result["message"]
            if isinstance(self.exec_result["message"], str) and len(self.exec_result["message"])==0:
                all += "\nNo output from your program. Use ```print(variable_value_you_want_to_know)``` to display the value of a variable."
            memory.append(LLMTPC_Memory(send_role, send_name, all, system_role="user"))
        else:   # program executed failed
            all = self.res_dict["debug_message"].content
            if "AssertionError" in self.exec_result["message"] or "TypeError" in self.exec_result["message"]:
                all = all.format(self.res_dict["debug_message"].function)
            else:
                all = all.format("")
            system_role = self.res_dict["debug_message"].system_role
            all = "Observation: " + self.exec_result["message"] + "\n" + all
            memory.append(LLMTPC_Memory(send_role, send_name, all, system_role=system_role))
    
    def _process_TPC_state_program_no_reflection(self, send_role, send_name, memory):
        all = "Observation: " + self.exec_result["message"]
        memory.append(LLMTPC_Memory(send_role, send_name, all, system_role="user"))

    def _process_TPC_state_final_answer(self, parse_res, memory):
        pass
    
    def _process_TPC_state_parse_error(self, parse_res, setting, memory):
        send_name = self.name
        send_role = self.role

        use_plan = setting["use_plan"] if "use_plan" in setting else True

        missing_attr = []
        missing_info = ""
        action_error = ""
        for info in ["Thought", "Action", "Action Input"]:
            if not parse_res[info]:
                if info == "Thought" and not use_plan:
                    continue
                missing_attr.append(info)
        if missing_attr:
            missing_info = f"Your response miss some information: {missing_attr}. "
        if parse_res["Action"] and parse_res["Action"] not in ["Program", "Final Answer"]:
            action_error = f"You chose the wrong action: {parse_res['Action']}. Action must be chosen from: [\"Program\", \"Final Answer\"]. "
        all = self.res_dict["parse_message"].content.format(missing_info + action_error)
        system_role = self.res_dict["parse_message"].system_role
        memory.append(LLMTPC_Memory(send_role, send_name, all, system_role=system_role))

    def _process(self):
        """
        processing action
        Rerutn : memory(List(Memory))
        """
        memory = []
        response = self.response
        send_name = self.name
        send_role = self.role
        if "new_chat_history" in self.res_dict:
            for mem in self.res_dict["new_chat_history"]:
                memory.append(LLMTPC_Memory(send_role, send_name, mem["content"], mem["role"]))
        if response:
            memory.append(LLMTPC_Memory(send_role, send_name, response, system_role="assistant"))
                    
        return memory


def parse_response(response):
    thought, action, action_input = None, None, None
    # Extract Thought
    match_thought = re.search(r"Thought:\s*(.*?)\nAction:", response, re.DOTALL)
    if match_thought:
        thought = match_thought.group(1)
    # Extract Action
    match_action = re.search(r"Action:\s+(.*?)\n", response)
    if match_action:
        action = match_action.group(1)
        response = response.replace("python", "Python")
        # Extract Action Input
        if action == "Program":
            match_action_input = re.search(r"```Python\n([\s\S]*?)\n```", response, re.DOTALL)
        else:
            match_action_input = re.search(r"Action Input:\s*(.*?)$", response, re.DOTALL)
        if match_action_input:
            action_input = match_action_input.group(1)
    else:
        match_action = re.search(r"Action:\s+(.*?)$", response, re.DOTALL)
        if match_action:
            action = match_action.group(1)
    return {"Thought": thought, "Action": action, "Action Input": action_input}
