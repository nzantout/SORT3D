import numpy as np
import re

class CodeParser:

    def __init__(self):
        """
        Module to parse code into waypoints
        """

    def go_near(self, obj, object_dict):
        full_obj = object_dict[obj]
        waypoint = np.array([float(full_obj["centroid"][0]), float(full_obj["centroid"][1])])
        return waypoint


    def go_between(self, obj1, obj2, object_dict):
        full_obj1 = object_dict[obj1]
        full_obj2 = object_dict[obj2]
        coord1 = np.array([float(full_obj1["centroid"][0]), float(full_obj1["centroid"][1])])
        coord2 = np.array([float(full_obj2["centroid"][0]), float(full_obj2["centroid"][1])])
        waypoint = (coord1 + coord2) / 2
        return waypoint


    def parse_code(self, code: str, object_dict: dict, object_id_map: dict):
        waypoints = []
        ids = []

        lines = code[code.find('Code:'):].split('\n')
        for line in lines:
            match = re.search(r"go_near\((.*?)\)", line)
            if match:
                obj = match.group(1)
                obj_num = re.findall(r'\d+', obj)[-1]
                print("obj num", obj, obj_num)
                try:
                    true_obj_id = object_id_map[int(obj)]
                    waypoints.append(self.go_near(true_obj_id, object_dict))
                    ids.append([true_obj_id])
                    continue
                except KeyError:
                    print("Not in object id map!")
                    print(object_id_map)
                    continue            
            match = re.search(r"go_between\((.*?), (.*?)\)", line)
            if match:
                obj1, obj2 = match.groups()
                try:
                    true_obj1_id = object_id_map[int(obj1)]
                    true_obj2_id = object_id_map[int(obj2)]
                    waypoints.append(self.go_between(true_obj1_id, true_obj2_id, object_dict))
                    ids.append([true_obj1_id, true_obj2_id])
                except KeyError:
                    print("Not in object id map!")
                    print(object_id_map)
                    continue

        
        waypoints = np.array(waypoints)
        if not len(waypoints) or not len(ids):
            print("No object parsed from code")
        return waypoints, ids
    
    def parse_code_from_tool(self, list_of_commands: list[tuple], object_dict: dict, object_id_map: dict):
        """
        Args:
            list_of_commands: list_of_commands, eg. [('go_near', (1,)), ('go_between', (2, 3))]")
        """
        Warning("Need an extra eye on this function!!!")
        waypoints = []
        ids = []
        for cmd, args in list_of_commands:
            if cmd == 'go_near':
                obj = args[0]
                true_obj_id = object_id_map[obj]
                waypoints.append(self.go_near(true_obj_id, object_dict))
                ids.append(true_obj_id)
            elif cmd == 'go_between':
                obj1, obj2 = args
                true_obj1_id = object_id_map[obj1]
                true_obj2_id = object_id_map[obj2]
                waypoints.append(self.go_between(true_obj1_id, true_obj2_id, object_dict))
                ids.append([true_obj1_id, true_obj2_id])
        return np.array(waypoints), ids

if __name__ == "__main__":
    code_parser = CodeParser()
    object_dict = {
        0: {"centroid": [0, 0, 0]},
        1: {"centroid": [1, 1, 0]},
        2: {"centroid": [2, 2, 0]},
        3: {"centroid": [3, 3, 0]},
    }
    object_id_map = {0: 0, 1: 1, 2: 2, 3: 3}
    code = '''
    Code:
    def go():
        go_near(0)
        go_near(1)
        go_between(2, 3)
    '''
    waypoints, ids = code_parser.parse_code(code, object_dict, object_id_map)
    print(waypoints)
    print(ids)
    print("Expected: [[0. 0.], [1. 1.], [2.5 2.5]]")
    print("Expected: [0, 1, [2, 3]]")
    print("Actual:", waypoints)
    print("Actual:", ids)

    list_of_commands = [('go_near', (0,)), ('go_near', (1,)), ('go_between', (2, 3))]
    waypoints2, ids2 = code_parser.parse_code_from_tool(list_of_commands, object_dict, object_id_map)
    print(waypoints2)
    print(ids2)
    print("Actual:", waypoints2)
    
    print("Equality check wp:", waypoints == waypoints2)
    print("Equality check id:", ids == ids2)
