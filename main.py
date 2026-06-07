from pulp import *

warehouses = ["W1", "W2", "W3"]
cities = ["C1", "C2", "C3", "C4"]

# Fixed cost to open each warehouse
opening_cost = {
    "W1": 400,
    "W2": 300,
    "W3": 350}

# Maximum capacity of each warehouse
capacity = {
    "W1": 100,
    "W2": 80,
    "W3": 90}

# Demand required by each city
demand = {
    "C1": 30,
    "C2": 40,
    "C3": 35,
    "C4": 25}

# Cost per unit shipped from warehouse to city
shipping_cost = {("W1", "C1"): 4, ("W1", "C2"): 6, ("W1", "C3"): 9, ("W1", "C4"): 5,
("W2", "C1"): 5, ("W2", "C2"): 4, ("W2", "C3"): 7, ("W2", "C4"): 6,
("W3", "C1"): 6, ("W3", "C2"): 5, ("W3", "C3"): 4, ("W3", "C4"): 7}

# Model Initialization
model = LpProblem("Warehouse_Location_and_Distribution", LpMinimize)

# Binary variable: if a warehouse is opened or not
open_warehouse = LpVariable.dicts(
    "OpenWarehouse",
    warehouses,
    lowBound=0,
    upBound=1,
    cat=LpBinary)

# Continuous variable: shipment amount
shipment = LpVariable.dicts(
    "Shipment",
    [(w, c) for w in warehouses for c in cities],
    lowBound=0,
    cat=LpContinuous)

# Objective Function:
# 1-) Fixed opening costs
# 2-) Variable shipping costs
model += (
    lpSum(opening_cost[w] * open_warehouse[w] for w in warehouses) +
    lpSum(shipping_cost[(w, c)] * shipment[(w, c)]
          for w in warehouses for c in cities))

# Constraints:
# Each city's total received shipment must equal its demand
for c in cities:
    model += lpSum(shipment[(w, c)] for w in warehouses) == demand[c]

# Warehouse capacity constraint
# If warehouse is closed, allowed shipment becomes zero
for w in warehouses:
    model += (
        lpSum(shipment[(w, c)] for c in cities)
        <= capacity[w] * open_warehouse[w]
    )

# Solve the Optimization
solver = PULP_CBC_CMD(msg=False)
status = model.solve(solver)

# Result Analysis Functions
def calculate_opening_cost():
    return sum(
        opening_cost[w] * value(open_warehouse[w])
        for w in warehouses
    )

def calculate_shipping_cost():
    return sum(
        shipping_cost[(w, c)] * value(shipment[(w, c)])
        for w in warehouses for c in cities
        if value(shipment[(w, c)]) is not None
    )

def warehouse_utilization(w):
    used = sum(value(shipment[(w, c)]) for c in cities)
    return used, capacity[w]

# Output Results
print("Optimization Status:", LpStatus[status])

print("\nOpened Warehouses:")
for w in warehouses:
    if value(open_warehouse[w]) > 0.5:
        print(f"  - {w}")

print("\nShipping Plan:")
total_shipped = 0
for w in warehouses:
    for c in cities:
        amount = value(shipment[(w, c)])
        if amount is not None and amount > 0:
            print(f"  {w} -> {c}: {amount}")
            total_shipped += amount

print("\nWarehouse Utilization:")
for w in warehouses:
    if value(open_warehouse[w]) > 0.5:
        used, cap = warehouse_utilization(w)
        print(f"  {w}: {used}/{cap}")

opening_total = calculate_opening_cost()
shipping_total = calculate_shipping_cost()

print("\nCost Breakdown:")
print("  Opening Cost:", opening_total)
print("  Shipping Cost:", shipping_total)
print("  Total Cost:", value(model.objective))
print("\nTotal Shipped Units:", total_shipped)
