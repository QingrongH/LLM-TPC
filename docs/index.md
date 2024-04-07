---
layout: project_page
permalink: /

title: "Think-Program-reCtify: 3D Situated Reasoning with Large Language Models"
authors:
    Qingrong He, Kejun Lin, Shizhe Chen, Anwen Hu and Qin Jin
affiliations:
    Renmin University of China, INRIA, Alibaba Group
paper: https://github.com/QingrongH/LLM-TPC
code: https://github.com/QingrongH/LLM-TPC
---

<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
This work addresses the 3D situated reasoning task which aims to answer questions given egocentric observations in a 3D environment.
The task remains challenging as it requires comprehensive 3D perception and complex reasoning skills. End-to-end models trained on supervised data for 3D situated reasoning suffer from data scarcity and generalization ability.
Inspired by the recent success of leveraging large language models (LLMs) for visual reasoning, we propose LLM-TPC, a novel framework that leverages the planning, tool usage, and reflection capabilities of LLMs through a <strong>T</strong>hink-<strong>P</strong>rogram-re<strong>C</strong>tify loop.  
The <strong>Think</strong> phase first decomposes the compositional question into a sequence of steps, and then the <strong>Program</strong> phase grounds each step to a piece of code and calls carefully designed 3D visual perception modules.
Finally, the <strong>Rectify</strong> phase adjusts the plan and code if the program fails to execute. 
Experiments and analysis on the SQA3D benchmark demonstrate the effectiveness, interpretability and robustness of our method.
        </div>
    </div>
</div>

---

## Overview

![Figure 1](/static/images/LLM-TPC.png)
*Figure 1: Overall Framework of LLM-TPC. LLM-TPC comprises three key components: the 3D Visual Perception Module equips the LLM with 3D context perception abilities, the Prompt Preparation Stage prepares prompts for reasoning, and the Reasoning Stage involves iterative Think-Program-reCtify loops.*

LLM-TPC contains a **T**hink-**P**rogram-re**C**tify loop to iteratively enhance the question answering performance.
In the ***Think*** phase, an LLM is prompted to decompose the question into a series of steps in natural language, taking advantage of LLM's world knowledge.
It then generates an executable Python program in the following ***Program*** phase guided by the steps in the Think phase. The program calls a set of 3D visual perception modules to query necessary information needed to solve the target question.
Next, in the ***Rectify*** phase, the program is executed and corrected if it fails or reaches a maximum number of iterations.
Finally, the final answer is formalized through summarizing the execution results. 


## Qualitative Results
![Figure 2](/static/images/qualitative_results_1.png)
*Figure 2: Qualitative results of LLM-TPC.*

LLM-TPC can leverage common-sense reasoning to provide knowledge-intensive answers that go beyond the information derived solely from the 3D scene. For example, it can determine whether it can reach the lamp from its current position by considering the typical arm length of a person (as shown in the top-left figure). It can also estimate the size of the bed based on its length and width (as shown in the top-right figure). Furthermore, it can make decisions about which furniture can be used for washing hands (as shown in the bottom figure).

![Figure 3](/static/images/qualitative_results_2.png)
*Figure 3: Qualitative results of LLM-TPC.*

When the execution fails, LLM-TPC enters the Rectify module, where the LLM is prompted with a debug command to loop back to the Think phase and adjusts the plan and program according to the received error message (as shown in the top figure). Additionally, we can detect the capabilities of APIs and foundational visual models by observing the intermediate output of program execution (as shown in the bottom figure).


## Citation
```
@article{qingrong2024llm-tpc,
  title={Think-Program-reCtify: 3D Situated Reasoning with Large Language Models},
  author={Qingrong, He and Kejun, Lin and Shizhe, Chen and Anwen, Hu and Qin, Jin},
  year={2024}
}
```
