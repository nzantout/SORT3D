example_object_list = [
    [0, 'coffee machine', 'a black coffee machine with a white and brown coffee cup on the left side', 65, 4, 8, 0.25],
    [1, 'chair', 'a black office chair with a silver pole in the middle of it.', 72, 43, 4, 0.3], # 5
    [2, 'chair', 'a black rolling office chair with a silver metal frame and black wheels.', 25, 52, 4, 0.3],
    [3, 'chair', 'a black rolling office chair that is facing left.', 26, 18, 4, 0.3],
    [4, 'chair', 'a gray metal chair with a gray metal pole in the middle of it.', 16, 44, 4, 0.35],
    [5, 'chair', 'the back of a black office chair placed on a brown hardwood', 82, 43, 4, 0.3], # 4
    [6, 'chair', 'a black office chair that is facing right.', 62, 25, 4, 0.3], # 1
    [7, 'chair', 'a black office chair placed on a brown wooden floor.', 82, 26, 4, 0.3], # 3
    [8, 'chair', 'a black leather office chair that is facing forward.', 34, 31, 4, 0.32],
    [9, 'chair', 'a black leather office chair with a silver metal base and black wheels.', 16, 29, 4, 0.37],
    [10, 'chair', 'a black office chair with a silver metal base and black wheels.', 34, 42, 4, 0.3],
    [11, 'chair', 'the back of a black office chair.', 72, 25, 4, 0.3], # 2
    [12, 'chair', 'three black leather office chairs placed side by side.', 62, 43, 4, 0.3], # 6
]

# 6, 11, 7, 5, 1, 12

example_object_list_without_captions = [obj[:2] + obj[3:] for obj in example_object_list]

def get_tool_call_example_old():
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
    example_object_list = [
    [0, 'coffee machine', 'a black coffee machine with a white and brown coffee cup on the left side', 65, 4, 8, 0.25],
    [1, 'chair', 'a black office chair with a silver pole in the middle of it.', 72, 43, 4, 0.3],
    [2, 'chair', 'a black rolling office chair with a silver metal frame and black wheels.', 25, 52, 4, 0.3],
    [3, 'chair', 'a black rolling office chair that is facing left.', 26, 18, 4, 0.3],
    [4, 'chair', 'a gray metal chair with a gray metal pole in the middle of it.', 16, 44, 4, 0.35],
    [5, 'chair', 'the back of a black office chair placed on a brown hardwood', 82, 43, 4, 0.3]
    ]
    human_msg = """"Object List:
    {example_object_list}
    User Input:
    Navigate to the chair closest to the coffee machine, then to the chair farthest from the coffee machine"""
    example_tool_calls = [
        HumanMessage(
            content=human_msg.format(example_object_list=example_object_list), name="example_user"
        ),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "notepad", "args": {"thoughts": "To navigate to the chair closest to the coffee machine, I need to find the distance between each of the chairs to the coffee machine. Then, I need to find the chair with the smallest distance to the coffee machine. Then, I will find the chair with the largest distance to the coffee machine."}, "id": "1"}
            ],
        ),
        ToolMessage("Noted and please proceed with the next step!", tool_call_id="1"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "calculate_distance", "args": {
                    "point1": (65, 4, 8),
                    "point1_name": "coffee machine",
                    "other_points": [(72, 43, 4), (25, 52, 4), (26, 18, 4), (16, 44, 4), (82, 43, 4)],
                    "other_points_names": ["chair 1", "chair 2", "chair 3", "chair 4", "chair 5"],
                }, "id": "2"}
            ],
        ),
        ToolMessage("""The distance between coffee machine at (65, 4, 8) and chair 1 at (72, 43, 4) is 39.8 units
The distance between coffee machine at (65, 4, 8) and chair 2 at (25, 52, 4) is 62.6 units
The distance between coffee machine at (65, 4, 8) and chair 3 at (26, 18, 4) is 41.6 units
The distance between coffee machine at (65, 4, 8) and chair 4 at (16, 44, 4) is 63.4 units
The distance between coffee machine at (65, 4, 8) and chair 5 at (82, 43, 4) is 42.7 units""",tool_call_id="2"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "notepad", "args": {"thoughts": "Now I need to call `find_min_max` to figure out which chair is the nearest and which chair is the farthest"}, "id": "3"}
            ],
        ),
        ToolMessage("Noted and please proceed with the next step!", tool_call_id="3"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "find_min_max", "args": {
                    "objects": ["chair 1", "chair 2", "chair 3", "chair 4", "chair 5"],
                    "attributes": [39.8, 62.6, 41.6, 63.4, 42.7],
                }, "id": "4"}
            ],
        ),
        ToolMessage("The object with the minimum value is chair 1 with value 39.8, and the object with the maximum value is chair 4 with value 63.4", tool_call_id="4"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "notepad", "args": {"thoughts": "I will now navigate to chair 1 and then to chair 4."}, "id": "5"}
            ],
        ),
        ToolMessage("Noted and please proceed with the next step!", tool_call_id="5"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "command_robot", "args": {"list_of_commands": [("go_near", (1,)), ("go_near", (4,))]}, "id": "6"}
            ],
        ),
        ToolMessage("The robot has executed the commands successfully, now say 'done'!", tool_call_id="6"),
        AIMessage(
            "done",
            name="example_assistant",
        ),
    ]
    return example_tool_calls


def get_tool_call_example_1(benchmark_mode=False):
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
    example_object_list = [
        [
            "0",
            "0",
            "window",
            "The window is brown, wood, rectangular, with a frame that appears to be made of wood. It has a glass pane that allows light to pass through, and it seems to be part of a larger structure, possibly a house or building. The window has a simple design with no visible decorations or markings.",
            "51",
            "0",
            "18",
            "0.9977728327159009"
        ],
        [
            "1",
            "1",
            "window",
            "The window is white, plastic, rectangular, and has a fixed position on the wall. It appears to be a standard window with no visible glass or panes. The window does not have any visible handles or mechanisms for opening or closing.",
            "65",
            "71",
            "15",
            "0.43602745120622316"
        ],
        [
            "3",
            "2",
            "kitchen counter",
            "kitchen counter",
            "65",
            "11",
            "9",
            "1.135686110039675"
        ],
        [
            "8",
            "3",
            "cabinet",
            "The cabinet is light brown, made of wood, and has a rectangular shape with doors and shelves. It has affinities for storing books and other items.",
            "8",
            "11",
            "11",
            "4.54341808202991"
        ],
        [
            "10",
            "4",
            "sink",
            "The sink is white, ceramic, rectangular",
            "67",
            "57",
            "4",
            "0.6072393512733472"
        ],
        [
            "33",
            "5",
            "refrigerator",
            "The refrigerator is white, made of metal, and has a rectangular shape with two doors. It has a handle on each door and is affixed to the wall. The refrigerator also has a freezer compartment at the top and a main fridge compartment below it.",
            "64",
            "23",
            "9",
            "1.9838631953428758"
        ],
        [
            "38",
            "6",
            "cabinet",
            "The cabinet is light brown, made of wood, and has a rectangular shape with two doors on each side. It has a classic design with a wooden frame around the top and a small handle on each door. The cabinet appears to be sturdy and well-crafted.",
            "67",
            "45",
            "11",
            "2.478599958926959"
        ],
        [
            "39",
            "7",
            "cabinet",
            "The cabinet is light brown, made of wood, rectangular with a flat top surface. It has a shelf below for storage and appears to be sturdy and functional.",
            "5",
            "65",
            "9",
            "2.3513523430237995"
        ],
        [
            "40",
            "8",
            "kitchen cabinets",
            "The kitchen cabinets are light brown, made of wood, and have a rectangular shape with a slightly curved top edge.",
            "63",
            "24",
            "20",
            "0.7559998358078653"
        ],
        [
            "41",
            "9",
            "cabinet",
            "The cabinet is yellow, wood, rectangular, affords storage",
            "45",
            "67",
            "18",
            "0.6493028297069858"
        ],
        [
            "42",
            "10",
            "cabinet",
            "The cabinet is light brown, wood, rectangular, and has drawers and doors for storage.",
            "62",
            "11",
            "4",
            "1.6363168466731803"
        ],
        [
            "43",
            "11",
            "kitchen cabinets",
            "The kitchen cabinets are light brown, made of wood, and rectangular in shape. They have a simple design with visible hinges and knobs. The cabinets appear to be made of solid wood or a wood composite material.",
            "65",
            "10",
            "18",
            "1.7783302277215447"
        ]
    ]
    human_msg = """"Object List:
    {example_object_list}
    User Input:
    Navigate to The windows in the kitchen."""
    example_tool_calls = [
        HumanMessage(
            content=human_msg.format(example_object_list=example_object_list), name="example_user"
        ),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "notepad", "args": {"thoughts": "I need to find the window in the kitchen. Before that, I need to pick an object from the list that is most likely to be found in a kitchen. A refrigerator is most likely to be found in the kitchen. Therefore, I can use the `find_near` tool to find the window that is closest to the refrigerator."}, "id": "1"}
            ],
        ),
        ToolMessage("Noted and please proceed with the next step!", tool_call_id="1"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "find_near", "args": {
                    "target_name": "window",
                    "anchor_id": 5,
                }, "id": "2"}
            ],
        ),
        ToolMessage("[0, 1]", tool_call_id="2"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "notepad", "args": {"thoughts": "The window that is closest to the refrigerator is window 0. Therefore, the statement refers to window 0. I will navigate to window 0."}, "id": "3"}
            ],
            # tool_calls=[
            #     {"name": "notepad", "args": {"thoughts": "The window that is closest to the refrigerator is window 0. I will repeat my reasoning with all the information I have been given to make sure my reasoning is correct. I first needed to find the window in the kitchen. I did that by considering the objects that are most likely to be in the kitchen. I chose the refrigerator as the object most likely to be in the kitchen. I then used the find_near function to find the window closest to the refrigerator, and deduced from the returned list [0, 1] that window 0 is closest to the refrigerator. This reasoning makes sense. Therefore, I conclude that the statement refers to window 0. I will pick window 0."}, "id": "3"}
            # ],
        ),
        ToolMessage("Noted and please proceed with the next step!", tool_call_id="3")
    ]
    if benchmark_mode:
        example_tool_calls.append(
            AIMessage(
                "",
                name="example_assistant",
                tool_calls=[
                    {"name": "command_robot", "args": {"list_of_commands": [("go_near", (0,))]}, "id": "4"}
                ],
            )
        )
    else:
        example_tool_calls.append(
            AIMessage(
                "",
                name="example_assistant",
                tool_calls=[
                    {"name": "pick_object", "args": {"object_id": 0}, "id": "4"}
                ],
            )
        )
    example_tool_calls.extend([
        ToolMessage("The robot has executed the commands successfully, now say 'done'!", tool_call_id="4"),
        AIMessage(
            "done",
            name="example_assistant",
        ),
    ])
    return example_tool_calls

def get_tool_call_example_2():
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
    example_object_list = [
        [
            "0",
            "0",
            "sofa",
            "The sofa is black, leather, modern."
            "51",
            "0",
            "18",
            "0.9977728327159009"
        ],
        [
            "1",
            "1",
            "coffee machine",
            "The coffee machine is black, expensive.",
            "65",
            "71",
            "15",
            "0.43602745120622316"
        ],
    ]
    human_msg = """"Object List:
    {example_object_list}
    User Input:
    I am sitting on the sofa, and my coffee is done brewing. Fetch it for me."""
    example_tool_calls = [
        HumanMessage(
            content=human_msg.format(example_object_list=example_object_list), name="example_user"
        ),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "notepad", "args": {"thoughts": "The user's coffee has finished brewing, and the user is sitting at the sofa. There is a coffee machine in the list of objects, so the coffee is most likely there. I will go to the coffee machine to fetch the coffee, then go to the sofa the user is sitting at."}, "id": "1"}
            ],
        ),
        ToolMessage("Noted and please proceed with the next step!", tool_call_id="1"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {"name": "command_robot", "args": {"list_of_commands": [("go_near", (1,)), ("go_near", (0,))]}, "id": "2"}
            ],
        ),
        ToolMessage("The robot has executed the commands successfully, now say 'done'!", tool_call_id="2"),
        AIMessage(
            "done",
            name="example_assistant",
        ),
    ]
    return example_tool_calls