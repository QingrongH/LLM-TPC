import api.api
from api.api import *
import sys
import traceback
from io import StringIO

code = """\
# Get object set in the scene
object_set = scene()
# Filter table
table_set = filter(object_set, "table")
# Find table that is on my left
table_on_my_left_set = relate_agent(table_set, "left")
# Filter chair
chair_set = filter(object_set, "chair")
# Find chair that is on the table (that is on my left)
chair_on_all_table_set = set()
for table in table_on_my_left_set:
    chair_on_one_table_set = relate(chair_set, table, "on")
    chair_on_all_table_set.update(chair_on_one_table_set)
print("The number of chairs on the table that is on my left is:", len(chair_on_all_table_set))
"""

def execute_program(program, scene_info={}, attr_matcher=None, setting={"use_caption":True, "use_openshape":True}, global_vars=globals()):
    exec_result = {"execution state": "SUCCESS", "message": ""}
    old_stdout = sys.stdout
    new_stdout = StringIO()
    sys.stdout = new_stdout

    api.api.scene_data = scene_info["scene"]
    api.api.match_sqa3d_attr = attr_matcher
    api.api.use_caption = setting["use_caption"]
    api.api.use_openshape = setting["use_openshape"]

    try:
        exec(program, global_vars)
    except Exception as e:
        error_message = traceback.format_exc().strip().split("\n")[-1]
        print(error_message)
        exec_result["execution state"] = "ERROR"

    output = new_stdout.getvalue()
    exec_result["message"] = output.strip()
    sys.stdout = old_stdout
    # print(exec_result["message"])
    return exec_result
