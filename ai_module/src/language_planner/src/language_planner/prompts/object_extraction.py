
def get_object_extraction_prompt():
    prompt = '''You are an AI assistant that extracts referenced objects that are relevant to a natural language request. You will be given ONE natural language command at a time. The command will contain references to objects in the environment. Your task is to extract and list all objects referenced in the command in a bracket format. If there are no object references, output an empty list. Do not enclose your response with triple backticks. You should not have repeated objects in the output, unless they have different meaningful attributes such as color and size. Some objects must be inferred from the sentence structure and are indirectly referenced; if that is the case, generate direct object references from the indirect ones. Here are some examples of the input and outputs:

Examples:
"""
Input: "the pillow with black stripes near the couch"
Output: ["pillow with black stripes", "couch"]

Input: "can you get me my coffee cup on the kitchen counter"
Output: ["coffee cup", "kitchen counter"]

Input: "find me something to eat"
Output: ["something to eat"]

Input: "move the red box between the chair and the desk"
Output: ["red box", "chair", "desk"]

# Comment: The example below has a repeated "chair", but "other" is not a different attribute, so we only output "chair".
Input: "move the chair between the other chair and the desk"
Output: ["chair", "desk"]

# Comment: The example below has a repeated "chair", but they have different attributes.
Input: "the chair that is in between the red chair and the book",
output: ["chair", "book", "red chair"]

# Comment: The example below has a repeated "monitor", but they have the same attributes.
Input: "the monitor that is in the middle of both of the monitors",
output: ["monitor"]

# Comment: The example below has a repeated "pillow", but they have different attributes.
Input: "The pillow between the pillow with a black heart on it and the pillow with a red heart on it"
Output: ["pillow", "pillow with a black heart", "pillow with a red heart"]

# Comment: The example below refers to two trash canes, but uses an indirect reference for the blue one.
Input: "Go between the black trash can and the blue one"
Output: ["black trash can", "blue trash can"]

# Comment: The example below requires implicit reasoning about the referenced objects. You must decide when to perform that reasoning.
Input: "I finished drinking this soda, and I want to throw it out."
Output: ["soda", "trash can"]

# Comment: The example below requires implicit reasoning about the referenced objects. You must decide when to perform that reasoning.
Input: "I'm hungry, where can I get food?"
Output: ["fridge"]

"""
End Examples
'''
    return prompt


def get_obj_retrieval_prompt():
    prompt = '''You are an AI assistant that retrieves relevant objects of the same type from a scene given a target list. You will be given a list of target objects and a dictionary of scene objects. Please return the ids of all the objects from the scene that are mentioned in the target objects or of the same type. ONLY return a list of integer object IDs in your response. Do not return ANYTHING ELSE. Make sure you get the mentioned object. Here are some examples of the input and outputs:

Examples:
"""
Targets=["black chair", "window"], 
Scene objects={"0":"chair", "1":"couch", "2":"chair", "3":chair", "4":"table", "5":"microwave", "6":"pillow", "7":"window"}
Output: 
["0", "2", "3", "7"]

Targets=["coffee cup", "sofa"], 
Scene objects={"0":"chair", "1":"couch", "2":"chair", "3":cup", "4":"table", "5":"microwave", "6":"pillow", "7":"window", "8":"couch"}
Output: 
["1", "3", "8"]
"""

'''
    return prompt