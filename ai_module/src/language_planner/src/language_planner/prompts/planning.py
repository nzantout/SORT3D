from .examples import example_object_list_without_captions

def get_prompt(environment_name, grid_map_shape, robot_coords, objects):

	objects_no_captions = [obj[:2] + obj[3:] for obj in objects]
    
	prompt = f'''You are an intelligent agent, tasked with generating Python code to move a wheeled robot around a 2D environment from natural language instructions.

The environment is {environment_name}. The environment is split into a discrete {grid_map_shape[0]}x{grid_map_shape[1]} grid. The robot starts at {tuple(robot_coords[0].tolist())}.

You are given a list of objects in the environment. Each object is characterized by an identifying number id, a name, and (x, y) coordinates of its center in the following format: [id, 'name', x, y]

The following Python functions are implemented. Objects are passed into each function by its identifying number id:

def go_between(object1, object2):
	"""
	Go between object1 and object2.
	"""
def go_near(object):
	"""
	Go near object.
	"""
    
After the list of objects in the environment, you are given a natural language command asking the robot to navigate around the environment, and you are tasked with generating a python function from this prompt that only uses the above functions. Use context clues from the environment to understand some of the commands. Use chain-of-thought reasoning, and explain your reasoning as you parse the language into code. Write out your reasoning, then write out the output code. Here are two examples of the input and output formats:

Example 1:

Object List:
{example_object_list_without_captions}

User Input:
Go to the coffee machine, first passing near all of the three chairs closest to it, in order from closest to your initial position to furthest.

Your Output:

Reasoning:
[Explain your reasoning here]

Code:
def go():
    go_near(6)
    go_near(11)
    go_near(7)
    go_near(0)

Example 2:

Object List:
{example_object_list_without_captions}

User Input:
There are six office workers on the table near the coffee machine, and they need their morning coffee. You can only hold one coffee cup at a time.

Your Output:

Reasoning:
[Explain your reasoning here]

Code:
def go():
	go_near(0)
	go_near(6)
	go_near(0)
	go_near(11)
	go_near(0)
	go_near(7)
	go_near(0)
	go_near(5)
	go_near(0)
	go_near(1)
	go_near(0)
	go_near(12)

Respond ONLY in the following format. Make sure you think step by step, and write out your reasoning, then the code as shown below.

Your Output:

Reasoning: 
[Thoroughly explain the reasoning behind your answer.]

Code:
def go():
    # Insert your output code here.

Given the following list of objects and the input language command, write a Python script that could be executed by the wheeled robot:

Object List:
{objects}
'''
	return prompt

