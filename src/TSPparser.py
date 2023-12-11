import numpy as np

def read_instance_from_file(file_path: str) -> None:
        tsp_data = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("NAME"):
                    tsp_data["NAME"] = line.split(":")[1].strip()
                elif line.startswith("TYPE"):
                    tsp_data["TYPE"] = line.split(":")[1].strip()
                elif line.startswith("COMMENT"):
                    tsp_data["COMMENT"] = line.split(":")[1].strip()
                elif line.startswith("DIMENSION"):
                    tsp_data["DIMENSION"] = int(line.split(":")[1].strip())
                    tsp_data["COORDS"] = np.zeros((tsp_data["DIMENSION"], 2))
                elif line.startswith("EDGE_WEIGHT_TYPE"):
                    tsp_data["EDGE_WEIGHT_TYPE"] = line.split(":")[1].strip()
                elif line.startswith("EDGE_WEIGHT_FORMAT"):
                    tsp_data["EDGE_WEIGHT_FORMAT"] = line.split(":")[1].strip()
                elif line.startswith("DISPLAY_DATA_TYPE"):
                    tsp_data["DISPLAY_DATA_TYPE"] = line.split(":")[1].strip()
                elif line == "NODE_COORD_SECTION":
                    tsp_data["NODE_COORD_SECTION"] = {}
                    for coord_line in file:
                        coord_line = coord_line.strip()
                        if coord_line == "EOF" or coord_line == "":
                            break
                        parts = coord_line.split()
                        node_id = int(parts[0])
                        x_coord = float(parts[1])
                        y_coord = float(parts[2])
                        tsp_data["COORDS"][node_id - 1] = [x_coord, y_coord]
        return tsp_data["COORDS"]