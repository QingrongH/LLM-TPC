import sys
sys.path.append("../src/LLMTPC_agent")

class LLMTPC_Memory:
    def __init__(self, send_role, send_name, content, system_role="system", system_name="") -> None:
        self.send_role = send_role
        self.send_name = send_name
        self.content = content
        self.system_role = system_role
        self.system_name = system_name
    
    def get_gpt_message(self,role):
        return {"role":role,"content":self.content}
    