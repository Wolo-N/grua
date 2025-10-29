import numpy as np

# Simulate the geometry creation from gruita.py
crane_length = 30.0
tower_height = 25.0
boom_height = 4.0
n_boom_segments = 10

boom_x = np.linspace(0, crane_length, n_boom_segments + 1)
boom_y_top = np.full(n_boom_segments + 1, tower_height + boom_height)
boom_y_bot = np.full(n_boom_segments + 1, tower_height)

total_nodes = 1 + 2 * (n_boom_segments + 1)
X = np.zeros([total_nodes, 2], float)

node_idx = 0

# Tower base
X[0] = [0, 35]
tower_base = 0
node_idx += 1

# Boom top chord nodes
boom_top_nodes = []
for i in range(n_boom_segments + 1):
    X[node_idx] = [boom_x[i], boom_y_top[i]]
    boom_top_nodes.append(node_idx)
    node_idx += 1

# Boom bottom chord nodes
boom_bot_nodes = []
for i in range(n_boom_segments + 1):
    X[node_idx] = [boom_x[i], boom_y_bot[i]]
    boom_bot_nodes.append(node_idx)
    node_idx += 1

print("Node positions:")
print(f"Node 0 (tower_base): {X[0]}")
print(f"Node 1 (boom_top_nodes[0]): {X[1]}")
print(f"Node 12 (boom_bot_nodes[0]): {X[12]}")
print()
print(f"tower_base = {tower_base}")
print(f"boom_top_nodes[0] = {boom_top_nodes[0]}")
print(f"boom_bot_nodes[0] = {boom_bot_nodes[0]}")
print()

# Now create connectivity
elements = []

# Tower element
elements.append([tower_base, boom_bot_nodes[0]])
print(f"Added tower element: [{tower_base}, {boom_bot_nodes[0]}]")

# Boom top chord elements
for i in range(len(boom_top_nodes) - 1):
    elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])
    if i < 2:
        print(f"Added boom top chord: [{boom_top_nodes[i]}, {boom_top_nodes[i+1]}]")

# Boom bottom chord elements
for i in range(len(boom_bot_nodes) - 1):
    elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])

# Boom vertical members
for i in range(len(boom_top_nodes)):
    elements.append([boom_top_nodes[i], boom_bot_nodes[i]])
    if i < 2:
        print(f"Added vertical member: [{boom_top_nodes[i]}, {boom_bot_nodes[i]}]")

# Boom diagonal members
for i in range(len(boom_top_nodes) - 1):
    if i % 2 == 0:
        elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
        if i < 2:
            print(f"Added diagonal (even): [{boom_top_nodes[i]}, {boom_bot_nodes[i+1]}]")
    else:
        elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
        if i < 2:
            print(f"Added diagonal (odd): [{boom_bot_nodes[i]}, {boom_top_nodes[i+1]}]")

# Support ties
elements.append([tower_base, boom_top_nodes[6]])
elements.append([tower_base, boom_top_nodes[10]])
print(f"Added support tie 1: [{tower_base}, {boom_top_nodes[6]}]")
print(f"Added support tie 2: [{tower_base}, {boom_top_nodes[10]}]")

print("\nElements involving nodes 0 or 1 BEFORE filtering:")
for i, elem in enumerate(elements):
    if 0 in elem or 1 in elem:
        print(f"  Element {i}: {elem}")

# Filter
filtered_elements = []
for elem in elements:
    if not ((elem[0] == 0 and elem[1] == 1) or (elem[0] == 1 and elem[1] == 0)):
        filtered_elements.append(elem)
    else:
        print(f"\nFILTERED OUT: {elem}")

print("\nElements involving nodes 0 or 1 AFTER filtering:")
for i, elem in enumerate(filtered_elements):
    if 0 in elem or 1 in elem:
        print(f"  Element {i}: {elem}")