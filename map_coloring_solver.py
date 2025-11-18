import pickle
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

# --- ✅ CSP class (for pickle loading)
class MapColoringCSP:
    def __init__(self, variables, domains, neighbors):
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors


# --- STEP 1: Load preprocessed data
try:
    with open("us_county_csp.pkl", "rb") as f:
        csp = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: 'us_county_csp.pkl' not found. Run the preprocessing script first.")
    exit()

# Check if domains were updated to 4 colors
sample_var = random.choice(csp.variables)
if len(csp.domains.get(sample_var, [])) < 4:
    print("⚠️ WARNING: CSP object has < 4 colors. Please re-run the updated preprocessing script.")

print(f"✅ Loaded CSP with {len(csp.variables)} total counties.")


# --- STEP 2: Select a Coherent Subset for Visualization
start_county = "Los Angeles County, CA" 
subset_size = 50 

# Build the subset by exploring neighbors up to a certain size (BFS-like approach)
subset_vars = set()
queue = [start_county]
while queue and len(subset_vars) < subset_size:
    county = queue.pop(0)
    if county not in subset_vars and county in csp.variables:
        subset_vars.add(county)
        # Add neighbors to the queue
        for neighbor in csp.neighbors.get(county, []):
            if neighbor not in subset_vars and neighbor in csp.variables:
                queue.append(neighbor)

subset_vars = list(subset_vars)[:subset_size]

# Filter neighbors and domains for the subset
subset_neighbors = {v: [n for n in csp.neighbors[v] if n in subset_vars] for v in subset_vars}
subset_domains = {v: csp.domains[v] for v in subset_vars}

if not subset_vars:
    print("❌ Error: Could not find any counties. Check your data or start_county name.")
    exit()
print(f"🧩 Using a cluster of {len(subset_vars)} related counties for visualization (starting from {start_county}).\n")


# --- STEP 3: MRV Heuristic
def select_unassigned_variable(assignment, domains):
    unassigned = [v for v in domains if v not in assignment]
    # MRV: Choose the variable with the Minimum Remaining Values (smallest domain)
    return min(unassigned, key=lambda var: len(domains[var])) if unassigned else None


# --- STEP 4: Forward Checking Algorithm
def forward_checking(csp, assignment, domains):
    if len(assignment) == len(domains):
        return assignment
    
    var = select_unassigned_variable(assignment, domains)
    if not var:
        return assignment # Should not happen if len(assignment) != len(domains)
    
    for value in domains[var]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        new_domains = deepcopy(domains)
        valid = True
        
        # Forward Checking: Prune domains of unassigned neighbors
        for neighbor in csp.neighbors.get(var, []):
            if neighbor in new_domains and neighbor not in new_assignment:
                if value in new_domains[neighbor]:
                    new_domains[neighbor].remove(value)
                    
                    # Constraint check: If neighbor's domain is empty, backtrack
                    if not new_domains[neighbor]:
                        valid = False
                        break
        
        if valid:
            result = forward_checking(csp, new_assignment, new_domains)
            if result:
                return result
                
    return None # Backtrack


# --- STEP 5: Solve using Forward Checking + MRV
print("🎨 Coloring using Forward Checking + MRV...")
subset_csp = MapColoringCSP(subset_vars, subset_domains, subset_neighbors)
solution = forward_checking(subset_csp, {}, subset_domains)

if not solution or len(solution) < len(subset_vars):
    print("❌ Coloring failed. This subset likely requires more than 4 colors, or the constraint problem is unsolvable.")
    exit()
else:
    print("✅ Coloring successful!")


# --- STEP 6: Enhanced Visualization (Focusing on clarity)
print("🖼️ Generating enhanced map visualization...")

# Build graph
G = nx.Graph()
for county, nbs in subset_neighbors.items():
    for nb in nbs:
        G.add_edge(county, nb)

# Define bright color palette
color_palette = {
    "Red": "#E63946",
    "Blue": "#457B9D",
    "Green": "#2A9D8F",
    "Yellow": "#F4A261",
    "gray": "#BDBDBD" 
}

# Assign colors (using solution from the Forward Checking)
node_colors = [color_palette.get(solution.get(node, "gray"), "#BDBDBD") for node in G.nodes]

# Use a tight spring layout to show the cluster
pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)

# Create figure
plt.figure(figsize=(12, 10))
plt.title("🗺️ Map Coloring of U.S. Counties (Coherent Cluster)",
             fontsize=16, fontweight="bold", pad=20)
plt.suptitle(
    f"Visualizing a cluster of {len(subset_vars)} counties starting from {start_county}.\nConstraint: No adjacent counties share the same color (4-Color Domain).",
    fontsize=11, y=0.93, color="dimgray")

# Draw edges (thinner and lighter)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#999999", width=0.5)

# Draw colored nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=250,
    edgecolors="black",
    linewidths=0.6
)

# Label all counties (show only the County Name)
labels = {n: n.split(',')[0] for n in G.nodes} 
nx.draw_networkx_labels(G, pos, labels=labels,
                         font_size=6, font_color="black")

# Legend (color meaning)
patches = [
    mpatches.Patch(color=c, label=f"{k} region") for k, c in color_palette.items()
]
# Only show the colors that were actually used
used_colors = set(solution.values())
patches = [p for p in patches if p.get_label().split()[0] in used_colors]

plt.legend(handles=patches, loc="upper right", fontsize=9, title="Color Regions Used")

# Info box
plt.text(-1.1, -1.05,
          f"Subset Counties: {len(subset_vars)}\nStart Region: {start_county}\nAlgorithm: Forward Checking + MRV\nColors Used: {len(used_colors)}",
          fontsize=9, color="black", bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.4"))

plt.axis("off")
plt.tight_layout()
plt.show()