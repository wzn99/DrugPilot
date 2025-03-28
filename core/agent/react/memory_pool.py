from llama_index.core.agent.react.types import ActionReasoningStep
import re, json

class MemoryPool():
    def __init__(self):
        self.arguments = {}
    
    def __str__(self):
        return f"MemoryPool(arguments={self.arguments.keys()})"
    
    def __repr__(self):
        return f"<MemoryPool(arguments={self.arguments})>"

    # Extract and save content marked with "Save:" from reasoning_step to memory pool
    def save_from_ActionReasoningStep(self, reasoning_step: ActionReasoningStep):
        # Try to match JSON string after "Save:"
        match = re.search(r"Save:\s*(\{.*?\})", reasoning_step.thought)
        
        # If no match found, return without processing
        if not match:
            return
        
        # If matched, extract and parse the JSON string
        save_str = match.group(1)
        extracted_args = json.loads(save_str)
        
        # Save extracted arguments to memory pool
        for k, v in extracted_args.items():
            if k in self.arguments:
                self.arguments[k].append(v)
            else:
                self.arguments[k] = [v]

    # Get parameters selected from memory pool
    def get_action_input(self, reasoning_step):
        valid = True
        action_input = reasoning_step.action_input
        for k, v in action_input.items():
            if isinstance(v, str) and v[0] == '(' and v[-1] == ')':  # Parameter selected from memory pool
                v = v.strip('()')
                # If parameter exists in memory pool, select the last element from its list
                if v in self.arguments.keys():
                    action_input[k] = self.arguments[v][-1]
                else:
                    valid = False
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], str) and v[0][0] == '(' and v[0][-1] == ')':
                v = v[0].strip('()')
                # If parameter exists in memory pool, select the last element from its list
                if v in self.arguments.keys():
                    action_input[k] = self.arguments[v][-1]
                else:
                    valid = False
        return action_input, valid

    # Save parameters extracted from the question
    def save_action_input(self, reasoning_step):
        action_input = reasoning_step.action_input
        for k, v in action_input.items():
            if isinstance(v, str) and v[0] == '(' and v[-1] == ')':  # Parameter selected from memory pool
                continue
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], str) and v[0][0] == '(' and v[0][-1] == ')':
                continue
            else:  # Parameter extracted from question
                if k == 'drug_smiles':
                    self.add_to_memory('user_smiles', v)
                elif k == 'property':
                    self.add_to_memory('user_property', v)
                elif k == 'cell_name':
                    self.add_to_memory('user_cell_line', v)
                else:
                    self.add_to_memory(k, v)
        return action_input
    
    # 1. Save parameters extracted from the question
    # 2. Get parameters selected from memory pool
    def get_and_save_action_input(self, reasoning_step):
        action_input = reasoning_step.action_input
        for k, v in action_input.items():
            if isinstance(v, str) and v[0] == '(' and v[-1] == ')':  # Parameter selected from memory pool
                v = v.strip('()')
                # If parameter exists in memory pool, select the last element from its list
                if v in self.arguments.keys():
                    action_input[k] = self.arguments[v][-1]
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], str) and v[0][0] == '(' and v[0][-1] == ')':
                v = v[0].strip('()')
                # If parameter exists in memory pool, select the last element from its list
                if v in self.arguments.keys():
                    action_input[k] = self.arguments[v][-1]
            else:  # Parameter extracted from question
                if k == 'drug_smiles':
                    self.add_to_memory('user_smiles', v)
                elif k == 'property':
                    self.add_to_memory('user_property', v)
                elif k == 'cell_name':
                    self.add_to_memory('user_cell_line', v)
        return action_input

    def save_observation(self, tool_name, observation_content):
        if tool_name == 'drug_cell_response_regression_generation':
            self.add_to_memory("generated_smiles", observation_content)
        elif tool_name == 'drug_cell_response_regression_optimization':
            self.add_to_memory("optimized_smiles", observation_content)
        else:
            self.add_to_memory(tool_name + "_result", observation_content)

    # Add a parameter to memory pool
    def add_to_memory(self, name, value):
        if name in self.arguments.keys():
            self.arguments[name].append(value)
        else:
            self.arguments[name] = [value]
        # print(f"Save:  {name}: {value}")

    # Clear memory pool
    def clear(self):
        self.arguments.clear()