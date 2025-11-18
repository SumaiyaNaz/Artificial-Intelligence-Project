import pandas as pd
import pickle

class MapColoringCSP:
    """Constraint Satisfaction Problem structure for Map Coloring."""
    def __init__(self, variables, domains, neighbors):
        self.variables = variables      # The regions/counties
        self.domains = domains          # var -> list of possible values (colors)
        self.neighbors = neighbors      # var -> list of neighbor vars


# --- Step 1: Load Dataset ---
file_path = "county_adjacency2010.csv" 
data = pd.read_csv(file_path)

print("Columns in dataset:", list(data.columns))
print("Total rows:", len(data))

# --- Step 2: Normalize column names ---
data.columns = data.columns.str.lower()

# --- Step 3: Extract Variables and Neighbors ---
variables = list(data['countyname'].unique())
neighbors = {}

for county in variables:
    # Get all rows where this county appears
    county_neighbors = data[data['countyname'] == county]['neighborname'].dropna().tolist()

    # Remove self if present (no county should be its own neighbor)
    county_neighbors = [n for n in county_neighbors if n != county]

    neighbors[county] = county_neighbors

# --- Step 4: Define Color Domains (FIXED: 4 Colors) ---
# Four colors are necessary to guarantee a solution for complex county maps.
colors = ['Red', 'Green', 'Blue', 'Yellow'] 
domains = {var: list(colors) for var in variables}

# --- Step 5: Create CSP Object ---
us_county_csp = MapColoringCSP(variables, domains, neighbors)

# --- Step 6: Print Preprocessing Output  ---
print("\n--- Preprocessing Output ---")
print("Total Regions (Counties):", len(us_county_csp.variables))
sample = us_county_csp.variables[0]
print("Sample Variable:", sample)
print("Neighbors of sample county:", us_county_csp.neighbors[sample])
print("Sample Domain (FIXED with 4 Colors):", us_county_csp.domains[sample])

# --- Step 7: Save Preprocessed Object ---
with open("us_county_csp.pkl", "wb") as f:
    pickle.dump(us_county_csp, f)

print("\n✅ Preprocessing complete! CSP object saved as 'us_county_csp.pkl' with 4 colors.")