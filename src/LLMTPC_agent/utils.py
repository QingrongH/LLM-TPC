# coding=utf-8
# Copyright 2023  The AIWaves Inc. team.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""helper functions for an LLM autonoumous agent"""
import random
import json
import datetime
import random
import os
from memory import LLMTPC_Memory as Memory

def save_logs(log_file, memory):
    directory = os.path.dirname(log_file)
    os.makedirs(directory, exist_ok=True)

    log = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            log = json.load(f)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    for mem in memory:
        if isinstance(mem, Memory):
            log.append({
                "timestamp": timestamp,
                "send_role": mem.send_role,
                "send_name": mem.send_name,
                "system_role": mem.system_role,
                "system_name": mem.system_name,
                "content": mem.content
            })
        else:
            log.append(mem)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
