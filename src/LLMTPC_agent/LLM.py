from abc import abstractclassmethod
import openai
import os
import time
import sys
sys.path.append("../src/LLMTPC_agent")
from memory import LLMTPC_Memory as Memory
from types import SimpleNamespace

def convert_dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = SimpleNamespace(**v)
            convert_dict_to_namespace(v)

class LLM:
    def __init__(self) -> None:
        pass

    @abstractclassmethod 
    def get_response():
        pass


class OpenAILLM(LLM):
    def __init__(self,**kwargs) -> None:
        super().__init__()

        openai.api_key = os.environ["API_KEY"]

        self.MAX_CHAT_HISTORY = eval(
            os.environ["MAX_CHAT_HISTORY"]) if "MAX_CHAT_HISTORY" in os.environ else 10
        
        self.model = kwargs["model"] if "model" in kwargs else "gpt-3.5-turbo-16k-0613"
        self.temperature = kwargs["temperature"] if "temperature" in  kwargs else 0.3
        self.log_path = kwargs["log_path"] if "log_path" in kwargs else "logs"
         
    def get_response(self,
                    chat_history,
                    system_prompt,
                    last_prompt=None,
                    stream=False,
                    functions=None,
                    function_call="auto",
                    WAIT_TIME=10,
                    **kwargs):
        """
        return LLM's response 
        """
        model = self.model
        temperature = self.temperature
    
        messages = []
        
        if chat_history:
            if len(chat_history) >  self.MAX_CHAT_HISTORY:
                chat_history = chat_history[- self.MAX_CHAT_HISTORY:]
            if isinstance(chat_history[0],dict):
                messages += chat_history
            elif isinstance(chat_history[0],Memory):
                messages += [memory.get_gpt_message(memory.system_role) for memory in chat_history]
        
        if kwargs["res_dict"]:
            if "new_chat_history" in kwargs["res_dict"]:
                messages += kwargs["res_dict"]["new_chat_history"]
        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=False)
                break
            except Exception as e:
                print(e)
                if "maximum context length is" in str(e):
                    self.model = "gpt-3.5-turbo-16k-0613"
                    model = self.model
                    messages = messages[:-1]
                    # assert False, "exceed max length"
                else:
                    print(f"Please wait {WAIT_TIME} seconds and resend later ...")
                    time.sleep(WAIT_TIME)

        if functions:
            return response.choices[0].message
        elif stream:
            return response["choices"][0]["message"]["content"]
        else:
            return response.choices[0].message["content"]
