import os
from tqdm import tqdm
import random
import re
import math

random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float

    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    #stats = augment_graph_features(line)
    fread.close()
    return stats



def augment_graph_features(properties):
    #properties = extract_numbers(text)
    num_nodes = int(properties[0])
    num_edges = properties[1]
    num_triangles = properties[3]
    
    # if num_nodes < 2 or num_edges < 0 or num_communities < 1:
    #     density = math.nan
    #     triangle_formation_ratio = math.nan
    #     edge_to_node_ratio = math.nan

    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))

    max_possible_triangles = math.comb(num_nodes, 3) if num_nodes >= 3 else 1
    triangle_formation_ratio = num_triangles / max_possible_triangles

    edge_to_node_ratio = num_edges / num_nodes

    augmented_feats = properties + [density, triangle_formation_ratio, edge_to_node_ratio]

    return augmented_feats