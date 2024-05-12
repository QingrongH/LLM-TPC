import json

class FewShotExampleComponent:
    def __init__(self, few_shot_example_path):
        few_shot_example = []
        with open(few_shot_example_path, 'r') as f:
            few_shot_example = json.load(f) # [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]
        self.few_shot_example = []
        for example in few_shot_example:    # [{}, {}, {}]
            self.few_shot_example.extend(example)

    def get_few_shot_example(self):
        return self.few_shot_example

class QueryComponent:
    def __init__(self, content, scene_info, use_context=False):
        self.system_role = "user"
        self.get_scene(content, scene_info, use_context)

    def get_scene(self, content, scene_info, use_context=False):
        scene = scene_info["scene"]
        situation = scene.situation
        question = scene.question
        object_num = scene.get_object_num()
        objects = []
        for obj, num in object_num.items():
            objects.append(str(num)+" "+obj)
        objects = ", ".join(objects)
        if use_context:
            context = scene.get_context()
            self.content = content.format(objects, context, situation, question)
        else:
            self.content = content.format(objects, situation, question)

    def get_message(self):
        return {"role":self.system_role, "content":self.content}

class StartMessageComponent:
    def __init__(self, content, system_role="system"):
        super().__init__()
        self.content = content
        self.system_role = system_role

    def get_message(self):
        return {"role":self.system_role, "content":self.content}

class LastMessageComponent:
    def __init__(self, content, system_role="system"):
        super().__init__()
        self.content = content
        self.system_role = system_role

    def get_message(self):
        return {"role":self.system_role, "content":self.content}

class DebugMessageComponent:
    def __init__(self, content, function, system_role="system"):
        super().__init__()
        self.content = content
        self.function = function
        self.system_role = system_role

    def get_message(self):
        return {"role":self.system_role, "content":self.content}

class ParseMessageComponent:
    def __init__(self, content, system_role="system"):
        super().__init__()
        self.content = content
        self.system_role = system_role

    def get_message(self):
        return {"role":self.system_role, "content":self.content}