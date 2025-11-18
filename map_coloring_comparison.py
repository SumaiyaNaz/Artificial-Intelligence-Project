import pickle
import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# --- ✅ CSP class (for pickle loading) ---
class MapColoringCSP:
    def __init__(self, variables, domains, neighbors):
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors

# --- STEP 1: Load and Prepare Data ---
try:
    with open("us_county_csp.pkl", "rb") as f:
        csp = pickle.load(f)
except FileNotFoundError:
    print("❌ Error: 'us_county_csp.pkl' not found.")
    exit()

# Set Subset Size to 30 for guaranteed fast comparison of all 4 models
start_county = "Los Angeles County, CA" 
subset_size = 30 

# Build the subset (BFS-like approach)
subset_vars = set()
queue = [start_county]
while queue and len(subset_vars) < subset_size:
    county = queue.pop(0)
    if county not in subset_vars and county in csp.variables:
        subset_vars.add(county)
        for neighbor in csp.neighbors.get(county, []):
            if neighbor not in subset_vars and neighbor in csp.variables:
                queue.append(neighbor)

subset_vars = list(subset_vars)[:subset_size]

# Filter data for the subset
subset_neighbors = {v: [n for n in csp.neighbors[v] if n in subset_vars] for v in subset_vars}
subset_domains = {v: csp.domains[v] for v in subset_vars}

# Train-Test Split (80/20)
random.seed(42) 
random.shuffle(subset_vars)
split_idx = int(0.8 * len(subset_vars))
train_vars = subset_vars[:split_idx]
test_vars = subset_vars[split_idx:]

train_domains = {k: v for k, v in subset_domains.items() if k in train_vars}
train_neighbors = {k: [n for n in subset_neighbors[k] if n in train_vars] for k in train_vars}
train_csp = MapColoringCSP(train_vars, train_domains, train_neighbors)

print(f"✅ Data Ready (Optimized): Subset={len(subset_vars)} | Training={len(train_vars)} | Testing={len(test_vars)}")
print("--------------------------------------------------")

# --------------------------------------------------
# --- CORE CSP FUNCTIONS (Algorithms & Heuristics) ---
# --------------------------------------------------

def backtracking_search(csp, assignment, domains, select_var_func, check_func):
    """Generic Backtracking search function."""
    if len(assignment) == len(domains):
        return assignment
    
    var = select_var_func(assignment, domains)
    if not var: return assignment
    
    for value in domains[var]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        
        # Apply constraint check 
        new_domains, valid = check_func(csp, var, value, new_assignment, deepcopy(domains))
        
        if valid:
            result = backtracking_search(csp, new_assignment, new_domains, select_var_func, check_func)
            if result:
                return result
    return None

def select_unassigned_mrv(assignment, domains):
    unassigned = [v for v in domains if v not in assignment]
    return min(unassigned, key=lambda var: len(domains[var])) if unassigned else None

def select_unassigned_sequential(assignment, domains):
    unassigned = [v for v in domains if v not in assignment]
    return unassigned[0] if unassigned else None

def simple_consistency_check(csp, var, value, assignment, domains):
    for neighbor in csp.neighbors.get(var, []):
        if neighbor in assignment and assignment[neighbor] == value:
            return domains, False
    return domains, True

def forward_checking_check(csp, var, value, assignment, domains):
    domains_copy = deepcopy(domains)
    valid = True
    for neighbor in csp.neighbors.get(var, []):
        if neighbor in domains_copy and neighbor not in assignment:
            if value in domains_copy[neighbor]:
                domains_copy[neighbor].remove(value)
                if not domains_copy[neighbor]:
                    valid = False
                    break
    return domains_copy, valid

def evaluate_model(solution, subset_neighbors, subset_domains, test_vars):
    if not solution: return 0.0
    correct = 0
    for county in test_vars:
        available_colors = list(subset_domains[county])
        for n in subset_neighbors.get(county, []):
            if n in solution and solution[n] in available_colors:
                available_colors.remove(solution[n])
        if available_colors:
            correct += 1
    accuracy = correct / len(test_vars) * 100
    return accuracy

# ---------------------------------------------
# --- STEP 2: Run and Compare All 4 Models ---
# ---------------------------------------------
models = [
    {"name": "1. Backtracking (BT)", "select_func": select_unassigned_sequential, "check_func": simple_consistency_check, "color": "#F44336"}, 
    {"name": "2. BT + MRV", "select_func": select_unassigned_mrv, "check_func": simple_consistency_check, "color": "#FFC107"},
    {"name": "3. Forward Checking (FC)", "select_func": select_unassigned_sequential, "check_func": forward_checking_check, "color": "#2196F3"},
    {"name": "4. FC + MRV (Best Combination)", "select_func": select_unassigned_mrv, "check_func": forward_checking_check, "color": "#4CAF50"}, 
]

results = []
best_time = float('inf')
best_model_name = ""
best_solution = None

print("📈 Starting Full 4-Model Comparison...")

for model in models:
    name = model["name"]
    start_time = time.perf_counter()
    
    solution = backtracking_search(
        train_csp, 
        {}, 
        train_domains, 
        model["select_func"], 
        model["check_func"]
    )
    
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000 # Time in milliseconds
    
    accuracy = 0.0
    if solution:
        accuracy = evaluate_model(solution, subset_neighbors, subset_domains, test_vars)
        if accuracy == 100.0 and elapsed_time < best_time:
            best_time = elapsed_time
            best_model_name = name
            best_solution = solution 

    results.append({
        "Model": name,
        "Time (ms)": elapsed_time,
        "Accuracy (%)": accuracy,
        "Color": model["color"],
        "Solution": solution
    })
    print(f"✅ {name}: Time={elapsed_time:.2f}ms, Accuracy={accuracy:.2f}%")

# Create DataFrame for easy plotting and table generation
results_df = pd.DataFrame(results)

# ----------------------------------------------------
# --- STEP 3: Generate Comparison BAR CHART (New Requirement) ---
# ----------------------------------------------------

# Sort by Time for coloring logic (lowest is best)
results_df['Time Rank'] = results_df['Time (ms)'].rank(method='min', ascending=True)

def get_color_by_rank(rank):
    """Assigns color based on rank: Lowest time (rank 1) is Green, Highest is Red."""
    if rank == 1:
        return '#4CAF50' # Green (Best)
    elif rank == 4:
        return '#F44336' # Red (Worst)
    elif rank == 2:
        return '#FFC107' # Amber
    else:
        return '#2196F3' # Blue

bar_colors = results_df['Time Rank'].apply(get_color_by_rank)

plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Time (ms)'], color=bar_colors)
plt.xlabel("CSP Algorithm Variant", fontsize=12)
plt.ylabel("Execution Time (ms)", fontsize=12)
plt.title("⏱️ Performance Comparison: Execution Time of 4 CSP Models", fontsize=14, fontweight='bold')
plt.xticks(rotation=15, ha='right')

# Add accuracy labels on top of the bars
for i, row in results_df.iterrows():
    plt.text(i, row['Time (ms)'] + max(results_df['Time (ms)']) * 0.02, 
             f"{row['Time (ms)']:.2f}ms\nAcc: {row['Accuracy (%)']:.2f}%", 
             ha='center', fontsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# --- STEP 4: Display Results Table and Map Graph ---
# ----------------------------------------------------
# Final Determination of Best Model and Solution for Graph
if best_model_name:
    highlight_text = f"🥇 BEST MODEL: {best_model_name} (Fastest Time: {best_time:.2f}ms with 100% Accuracy)"
else:
    best_model_row = results_df.loc[results_df['Accuracy (%)'].idxmax()]
    best_model_name = best_model_row['Model']
    best_solution = next(m["Solution"] for m in results if m["Model"] == best_model_name)
    highlight_text = f"⚠️ Could not find 100% accurate model. Displaying top model: {best_model_name}"

print("\n\n" + "="*80)
print(highlight_text)
print("="*80)
print("\n📊 RESULTS TABLE (Comparison of CSP Solvers):")
print(results_df[['Model', 'Time (ms)', 'Accuracy (%)']].to_markdown(index=False, floatfmt=".2f")) 
print("\n" + "="*80)

# Generate Map Coloring Graph (Visual check of the best solution)
if not best_solution:
     print("\n❌ Cannot generate map graph: No solution found for the best model.")
     exit()

print(f"\n🖼️ Generating Map Coloring Graph using coloring from {best_model_name}...")

G = nx.Graph()
for county, nbs in subset_neighbors.items():
    for nb in nbs:
        G.add_edge(county, nb)

color_palette = {
    "Red": "#E63946", "Blue": "#457B9D", "Green": "#2A9D8F", "Yellow": "#F4A261", "gray": "#BDBDBD" 
}
domain_colors = ['Red', 'Blue', 'Green', 'Yellow'] 
node_colors = [color_palette.get(best_solution.get(node, "gray"), "#BDBDBD") for node in G.nodes]
pos = nx.spring_layout(G, seed=42, k=0.1, iterations=50)

plt.figure(figsize=(14, 10))
plt.title("🗺️ U.S. County Map Coloring (Solved by Best Model)", fontsize=16, fontweight="bold", pad=20)
plt.suptitle(
    f"Coloring Solved by: {best_model_name} | Subset of {len(subset_vars)} Counties.",
    fontsize=11, y=0.93, color="dimgray")

nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#999999", width=0.5)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=350, edgecolors="black", linewidths=0.6)

labels_dict = {n: n.split(',')[0] for n in G.nodes}
nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=7, font_color="black")

patches_map = [mpatches.Patch(color=color_palette[c], label=f"{c} Region") for c in domain_colors]
plt.legend(handles=patches_map, loc="upper right", fontsize=9, title="4 Color Domains")

plt.axis("off")
plt.tight_layout()
plt.show()