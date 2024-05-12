import sys
sys.path.append("../src/LLMTPC_agent")
from component import FewShotExampleComponent, QueryComponent, LastMessageComponent, StartMessageComponent, DebugMessageComponent, ParseMessageComponent

class LLMTPC_State:
    def __init__(self, **kwargs):
        self.next_states = {}
        self.name = kwargs["name"]
        self.scene_info = kwargs["scene_info"]

        self.environment_prompt = (
            kwargs["environment_prompt"] if "environment_prompt" in kwargs else ""
        )

        self.roles = kwargs["roles"] if "roles" in kwargs else (list(kwargs["agent_states"].keys()) if "agent_states" in kwargs else [0])
        if len(self.roles) == 0:
            self.roles = [0]
        self.begin_role = (
            kwargs["begin_role"] if "begin_role" in kwargs else self.roles[0]
        )
        self.begin_query = kwargs["begin_query"] if "begin_query" in kwargs else None

        self.is_begin = True

        self.summary_prompt = (
            kwargs["summary_prompt"] if "summary_prompt" in kwargs else None
        )
        self.current_role = self.begin_role
        self.components = (
            self.init_components(kwargs["agent_states"])
            if "agent_states" in kwargs
            else {}
        )
        self.index = (
            self.roles.index(self.begin_role) if self.begin_role in self.roles else 0
        )
        self.chat_nums = 0

    def init_components(self, agent_states_dict: dict):
        agent_states = {}
        for role, components in agent_states_dict.items():
            component_dict = {}
            for component, component_args in components.items():
                if component:
                    if component == "few_shot_example":
                        few_shot_example_path = component_args["few_shot_example_path"]
                        component_dict[component] = FewShotExampleComponent(few_shot_example_path)
                    elif component == "query":
                        # scene_info = component_args["scene_info"]
                        content = component_args["content"]
                        use_context = component_args["use_context"] if "use_context" in component_args else False
                        scene_info = self.scene_info
                        component_dict[component] = QueryComponent(content, scene_info, use_context)
                    elif component == "last_message":
                        content = component_args["content"]
                        system_role = component_args["system_role"]
                        component_dict[component] = LastMessageComponent(content, system_role)
                    elif component == "start_message":
                        content = component_args["content"]
                        system_role = component_args["system_role"]
                        component_dict[component] = StartMessageComponent(content, system_role)
                    elif component == "debug_message":
                        content = component_args["content"]
                        function = component_args["function"]
                        system_role = component_args["system_role"]
                        component_dict[component] = DebugMessageComponent(content, function, system_role)
                    elif component == "parse_message":
                        content = component_args["content"]
                        system_role = component_args["system_role"]
                        component_dict[component] = ParseMessageComponent(content, system_role)
                    # ====================================================
                    else:
                        continue

            agent_states[role] = component_dict

        return agent_states
