import pandas as pd
import numpy as np
from functools import lru_cache
import os
import time
import pulp  # For Integer Programming


import streamlit as st

st.title("Hello from Colab!")
st.write("This is a simple Streamlit app running on Colab.")

class PackingOptimizer:
    def __init__(self, excel_file_path, cabin_weight_limit, cabin_volume_limit,
                 checkin_weight_limit, checkin_volume_limit, packer_cost_per_liter,
                 value_weight_ratio=0.5, value_volume_ratio=0.5):
        """
        Initialize the packing optimizer with constraints and item data from Excel file.

        Parameters:
        -----------
        excel_file_path : str
            Path to the Excel file containing item details
        cabin_weight_limit : float
            Maximum weight allowed in cabin baggage (kg)
        cabin_volume_limit : float
            Maximum volume allowed in cabin baggage (liters)
        checkin_weight_limit : float
            Maximum weight allowed in check-in baggage (kg)
        checkin_volume_limit : float
            Maximum volume allowed in check-in baggage (liters)
        packer_cost_per_liter : float
            Cost per liter for packers and movers (INR)
        value_weight_ratio : float
            Importance of value-per-weight in optimization (0.0 to 1.0)
        value_volume_ratio : float
            Importance of value-per-volume in optimization (0.0 to 1.0)
        """
        # Load data from Excel file
        self.items_data = pd.read_excel(excel_file_path)

        # Process the data into our format
        self.items = self._preprocess_items(self.items_data)

        # Calculate value-per-weight and value-per-volume for each item
        for item in self.items:
            item['value_per_weight'] = item['value'] / max(0.1, item['weight'])  # Avoid division by zero
            item['value_per_volume'] = item['value'] / max(0.1, item['volume'])  # Avoid division by zero

            # Calculate combined efficiency score
            item['efficiency_score'] = (
                value_weight_ratio * item['value_per_weight'] +
                value_volume_ratio * item['value_per_volume']
            )

        # Store importance ratios
        self.value_weight_ratio = value_weight_ratio
        self.value_volume_ratio = value_volume_ratio

        # Store constraints
        self.cabin_weight_limit = cabin_weight_limit
        self.cabin_volume_limit = cabin_volume_limit
        self.checkin_weight_limit = checkin_weight_limit
        self.checkin_volume_limit = checkin_volume_limit
        self.packer_cost_per_liter = packer_cost_per_liter

        # We'll use discretization for weight and volume to make the state space manageable
        # Each unit represents a fixed amount of weight/volume
        self.weight_unit = 0.5  # 0.5 kg per unit
        self.volume_unit = 1.0  # 1.0 liter per unit

        # Convert limits to discrete units
        self.cabin_weight_units = int(self.cabin_weight_limit / self.weight_unit)
        self.cabin_volume_units = int(self.cabin_volume_limit / self.volume_unit)
        self.checkin_weight_units = int(self.checkin_weight_limit / self.weight_unit)
        self.checkin_volume_units = int(self.checkin_volume_limit / self.volume_unit)

        # Total number of items
        self.n_items = len(self.items)

        # Memoization dictionary to store computed results
        self.memo = {}

        # To track the placement of each item for the optimal solution
        self.placement = {}

        # For tracking solution details in DP
        self.best_placement_cache = {}

    def _preprocess_items(self, items_data):
        """
        Preprocess items data into a list of dictionaries with required fields.
        """
        items = []

        # Determine column names based on the Excel file structure
        # This handles both "Item" and "ItemName" column naming conventions
        name_col = next((col for col in items_data.columns if 'item' in col.lower()), None)

        # If no explicit name column, try to use the first column
        if name_col is None and not items_data.empty:
            name_col = items_data.columns[0]

        # Define column mappings
        col_mapping = {
            'name': name_col,
            'weight': 'Weight (kg)',
            'volume': 'Volume (liters)',
            'value': 'Current Monetary Value (INR)',
            'baggage_type': 'Airline Baggage Type'
        }

        for idx, row in items_data.iterrows():
            # Extract item name
            if col_mapping['name'] and col_mapping['name'] in row:
                item_name = row[col_mapping['name']]
            else:
                item_name = f'Item_{idx}'

            # Create item dictionary with default values
            item = {
                'name': item_name,
                'weight': row.get(col_mapping['weight'], 0),
                'volume': row.get(col_mapping['volume'], 0),
                'value': row.get(col_mapping['value'], 0),
                'baggage_type': row.get(col_mapping['baggage_type'], 'Unknown'),
                'id': idx
            }

            # Set compatibility flags - fixed to handle "Both" baggage type
            item['cabin_compatible'] = item['baggage_type'] in ['Hand Baggage', 'Both']
            item['checkin_compatible'] = item['baggage_type'] in ['Check-in Baggage', 'Both']
            item['packer_compatible'] = True  # Assume all items can be sent via packers by default

            items.append(item)

        return items

    def optimize_with_dp(self):
        """
        Run the dynamic programming optimization that balances packer costs,
        value-per-weight, and value-per-volume efficiency.
        """
        # Convert item weights and volumes to discrete units
        for item in self.items:
            item['weight_units'] = max(1, int(np.ceil(item['weight'] / self.weight_unit)))
            item['volume_units'] = max(1, int(np.ceil(item['volume'] / self.volume_unit)))

        # Clear any previous placements and memoization
        self.placement = {}
        self.memo = {}
        self.best_placement_cache = {}

        # Start the recursion from the first item with empty baggage
        optimal_value = self._optimal_value_dp(0, 0, 0, 0, 0)

        # Reconstruct the solution using our placement cache
        self._reconstruct_solution(0, 0, 0, 0, 0)

        # Extract items for each baggage type
        cabin_items = []
        checkin_items = []
        packer_items = []

        # Create item lookup by ID
        item_lookup = {item['id']: item for item in self.items}

        for item_id, placement in self.placement.items():
            item = item_lookup.get(item_id)
            if item:
                if placement == 'cabin':
                    cabin_items.append(item)
                elif placement == 'checkin':
                    checkin_items.append(item)
                elif placement == 'packer':
                    packer_items.append(item)

        # Calculate baggage weights, volumes, and values
        cabin_weight = sum(item['weight'] for item in cabin_items)
        cabin_volume = sum(item['volume'] for item in cabin_items)
        checkin_weight = sum(item['weight'] for item in checkin_items)
        checkin_volume = sum(item['volume'] for item in checkin_items)
        packer_volume = sum(item['volume'] for item in packer_items)
        packer_cost = packer_volume * self.packer_cost_per_liter

        # Calculate total value for each section
        cabin_value = sum(item['value'] for item in cabin_items)
        checkin_value = sum(item['value'] for item in checkin_items)
        packer_value = sum(item['value'] for item in packer_items)
        total_value = cabin_value + checkin_value + packer_value

        # Calculate efficiency metrics
        cabin_value_per_weight = cabin_value / max(0.1, cabin_weight) if cabin_items else 0
        cabin_value_per_volume = cabin_value / max(0.1, cabin_volume) if cabin_items else 0
        checkin_value_per_weight = checkin_value / max(0.1, checkin_weight) if checkin_items else 0
        checkin_value_per_volume = checkin_value / max(0.1, checkin_volume) if checkin_items else 0

        return {
            'total_cost': packer_cost,
            'total_value': total_value,
            'cabin_items': cabin_items,
            'checkin_items': checkin_items,
            'packer_items': packer_items,
            'cabin_weight': cabin_weight,
            'cabin_volume': cabin_volume,
            'cabin_value': cabin_value,
            'cabin_value_per_weight': cabin_value_per_weight,
            'cabin_value_per_volume': cabin_value_per_volume,
            'checkin_weight': checkin_weight,
            'checkin_volume': checkin_volume,
            'checkin_value': checkin_value,
            'checkin_value_per_weight': checkin_value_per_weight,
            'checkin_value_per_volume': checkin_value_per_volume,
            'packer_volume': packer_volume,
            'packer_value': packer_value,
            'packer_cost': packer_cost
        }

    def _optimal_value_dp(self, item_idx, cabin_w, cabin_v, checkin_w, checkin_v):
        """
        Recursive DP function with memoization to find optimal packing plan.
        This uses a combined objective that balances minimizing packer costs and
        maximizing value-per-weight and value-per-volume efficiency.
        """
        # Base case: all items considered
        if item_idx >= self.n_items:
            return 0

        # Check if we've already computed this state
        state = (item_idx, cabin_w, cabin_v, checkin_w, checkin_v)
        if state in self.memo:
            return self.memo[state]

        current_item = self.items[item_idx]
        item_weight_units = current_item['weight_units']
        item_volume_units = current_item['volume_units']

        # Initialize with a large negative value (as we're maximizing)
        best_value = float('-inf')
        best_placement = None

        # Define a "benefit" function for each placement option
        # Higher is better, combining item value and efficiency

        # Option 1: Try cabin if compatible and within limits
        if (current_item['cabin_compatible'] and
                cabin_w + item_weight_units <= self.cabin_weight_units and
                cabin_v + item_volume_units <= self.cabin_volume_units):

            # Consider both item value and efficiency score for cabin placement
            cabin_benefit = current_item['value'] + 500 * current_item['efficiency_score']

            future_value = self._optimal_value_dp(
                item_idx + 1,
                cabin_w + item_weight_units,
                cabin_v + item_volume_units,
                checkin_w,
                checkin_v
            )

            option1_value = cabin_benefit + future_value

            if option1_value > best_value:
                best_value = option1_value
                best_placement = 'cabin'

        # Option 2: Try check-in if compatible and within limits
        if (current_item['checkin_compatible'] and
                checkin_w + item_weight_units <= self.checkin_weight_units and
                checkin_v + item_volume_units <= self.checkin_volume_units):

            # Consider both item value and efficiency score for checkin placement
            checkin_benefit = current_item['value'] + 300 * current_item['efficiency_score']

            future_value = self._optimal_value_dp(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w + item_weight_units,
                checkin_v + item_volume_units
            )

            option2_value = checkin_benefit + future_value

            if option2_value > best_value:
                best_value = option2_value
                best_placement = 'checkin'

        # Option 3: Send via packers
        if current_item['packer_compatible']:
            # Apply a penalty for using packers based on cost
            packer_penalty = current_item['volume'] * self.packer_cost_per_liter

            future_value = self._optimal_value_dp(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w,
                checkin_v
            )

            # We still get the item value but with a cost penalty
            option3_value = current_item['value'] - packer_penalty + future_value

            if option3_value > best_value:
                best_value = option3_value
                best_placement = 'packer'

        # Store the best placement for this state
        self.best_placement_cache[state] = best_placement

        # Memoize and return
        self.memo[state] = best_value
        return best_value

    def _reconstruct_solution(self, item_idx, cabin_w, cabin_v, checkin_w, checkin_v):
        """
        Reconstruct the solution by following the choices made during DP.
        """
        if item_idx >= self.n_items:
            return

        state = (item_idx, cabin_w, cabin_v, checkin_w, checkin_v)
        if state not in self.best_placement_cache:
            return

        placement = self.best_placement_cache[state]
        current_item = self.items[item_idx]

        self.placement[current_item['id']] = placement

        if placement == 'cabin':
            self._reconstruct_solution(
                item_idx + 1,
                cabin_w + current_item['weight_units'],
                cabin_v + current_item['volume_units'],
                checkin_w,
                checkin_v
            )
        elif placement == 'checkin':
            self._reconstruct_solution(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w + current_item['weight_units'],
                checkin_v + current_item['volume_units']
            )
        else:  # packer
            self._reconstruct_solution(
                item_idx + 1,
                cabin_w,
                cabin_v,
                checkin_w,
                checkin_v
            )

    def optimize_with_ip(self):
        """
        Run the optimization using Integer Programming (IP) approach.
        This method balances minimizing packer costs and maximizing value efficiency.
        """
        # Create a new LP problem
        model = pulp.LpProblem(name="Baggage_Packing", sense=pulp.LpMaximize)

        # Create binary decision variables for each item and baggage type
        x_cabin = {i: pulp.LpVariable(f"x_cabin_{i}", cat=pulp.LpBinary)
                  for i in range(self.n_items)}

        x_checkin = {i: pulp.LpVariable(f"x_checkin_{i}", cat=pulp.LpBinary)
                    for i in range(self.n_items)}

        x_packer = {i: pulp.LpVariable(f"x_packer_{i}", cat=pulp.LpBinary)
                   for i in range(self.n_items)}

        # Define the objective function:
        # 1. Maximize total value
        # 2. Consider value-per-weight and value-per-volume efficiency
        # 3. Minimize packer costs

        # Efficiency bonuses for cabin and check-in items
        cabin_efficiency_bonus = pulp.lpSum([
            (500 * self.items[i]['efficiency_score'] * x_cabin[i])
            for i in range(self.n_items)
        ])

        checkin_efficiency_bonus = pulp.lpSum([
            (300 * self.items[i]['efficiency_score'] * x_checkin[i])
            for i in range(self.n_items)
        ])

        # Packer cost penalty
        packer_cost_penalty = pulp.lpSum([
            self.items[i]['volume'] * self.packer_cost_per_liter * x_packer[i]
            for i in range(self.n_items)
        ])

        # Total item value
        total_value = pulp.lpSum([
            self.items[i]['value'] * (x_cabin[i] + x_checkin[i] + x_packer[i])
            for i in range(self.n_items)
        ])

        # Combined objective function
        model += total_value + cabin_efficiency_bonus + checkin_efficiency_bonus - packer_cost_penalty

        # Constraint 1: Each item must be placed somewhere
        for i in range(self.n_items):
            model += x_cabin[i] + x_checkin[i] + x_packer[i] == 1

        # Constraint 2: Cabin weight limit
        model += pulp.lpSum([self.items[i]['weight'] * x_cabin[i]
                             for i in range(self.n_items)]) <= self.cabin_weight_limit

        # Constraint 3: Cabin volume limit
        model += pulp.lpSum([self.items[i]['volume'] * x_cabin[i]
                             for i in range(self.n_items)]) <= self.cabin_volume_limit

        # Constraint 4: Check-in weight limit
        model += pulp.lpSum([self.items[i]['weight'] * x_checkin[i]
                             for i in range(self.n_items)]) <= self.checkin_weight_limit

        # Constraint 5: Check-in volume limit
        model += pulp.lpSum([self.items[i]['volume'] * x_checkin[i]
                             for i in range(self.n_items)]) <= self.checkin_volume_limit

        # Constraint 6: Cabin compatibility
        for i in range(self.n_items):
            if not self.items[i]['cabin_compatible']:
                model += x_cabin[i] == 0

        # Constraint 7: Check-in compatibility
        for i in range(self.n_items):
            if not self.items[i]['checkin_compatible']:
                model += x_checkin[i] == 0

        # Constraint 8: Packer compatibility
        for i in range(self.n_items):
            if not self.items[i]['packer_compatible']:
                model += x_packer[i] == 0

        # Solve the model
        model.solve(pulp.PULP_CBC_CMD(msg=False))

        # Extract the solution
        cabin_items = []
        checkin_items = []
        packer_items = []

        for i in range(self.n_items):
            if pulp.value(x_cabin[i]) == 1:
                cabin_items.append(self.items[i])
                self.placement[self.items[i]['id']] = 'cabin'
            elif pulp.value(x_checkin[i]) == 1:
                checkin_items.append(self.items[i])
                self.placement[self.items[i]['id']] = 'checkin'
            elif pulp.value(x_packer[i]) == 1:
                packer_items.append(self.items[i])
                self.placement[self.items[i]['id']] = 'packer'

        # Calculate total weights, volumes, values, and costs
        cabin_weight = sum(item['weight'] for item in cabin_items)
        cabin_volume = sum(item['volume'] for item in cabin_items)
        checkin_weight = sum(item['weight'] for item in checkin_items)
        checkin_volume = sum(item['volume'] for item in checkin_items)
        packer_volume = sum(item['volume'] for item in packer_items)
        packer_cost = packer_volume * self.packer_cost_per_liter

        cabin_value = sum(item['value'] for item in cabin_items)
        checkin_value = sum(item['value'] for item in checkin_items)
        packer_value = sum(item['value'] for item in packer_items)
        total_value = cabin_value + checkin_value + packer_value

        # Calculate efficiency metrics
        cabin_value_per_weight = cabin_value / max(0.1, cabin_weight) if cabin_items else 0
        cabin_value_per_volume = cabin_value / max(0.1, cabin_volume) if cabin_items else 0
        checkin_value_per_weight = checkin_value / max(0.1, checkin_weight) if checkin_items else 0
        checkin_value_per_volume = checkin_value / max(0.1, checkin_volume) if checkin_items else 0

        return {
            'total_cost': packer_cost,
            'total_value': total_value,
            'cabin_items': cabin_items,
            'checkin_items': checkin_items,
            'packer_items': packer_items,
            'cabin_weight': cabin_weight,
            'cabin_volume': cabin_volume,
            'cabin_value': cabin_value,
            'cabin_value_per_weight': cabin_value_per_weight,
            'cabin_value_per_volume': cabin_value_per_volume,
            'checkin_weight': checkin_weight,
            'checkin_volume': checkin_volume,
            'checkin_value': checkin_value,
            'checkin_value_per_weight': checkin_value_per_weight,
            'checkin_value_per_volume': checkin_value_per_volume,
            'packer_volume': packer_volume,
            'packer_value': packer_value,
            'packer_cost': packer_cost
        }


# Usage with Excel file
if __name__ == "__main__":
    # File path to the Excel file
    excel_file_path = "Indian_College_Student_Dorm_Items_Expanded (1).xlsx"

    # Check if file exists
    if not os.path.exists(excel_file_path):
        print(f"Error: Excel file '{excel_file_path}' not found.")
        exit(1)

    # Set constraints based on the provided Amazon link
    cabin_weight_limit = 7  # kg
    cabin_volume_limit = 44  # liters
    checkin_weight_limit = 25  # kg
    checkin_volume_limit = 116.13  # liters
    packer_cost = 50  # INR per liter

    # Value efficiency weighting parameters
    value_weight_ratio = 0.6  # 60% weight given to value-per-weight ratio
    value_volume_ratio = 0.4  # 40% weight given to value-per-volume ratio

    print(f"Loading data from {excel_file_path}...")

    # Create optimizer with efficiency parameters
    optimizer = PackingOptimizer(
        excel_file_path,
        cabin_weight_limit,
        cabin_volume_limit,
        checkin_weight_limit,
        checkin_volume_limit,
        packer_cost,
        value_weight_ratio,
        value_volume_ratio
    )

    print(f"Loaded {optimizer.n_items} items from the Excel file.")

    # Run and time the IP-based optimization
    print("\nRunning optimization with Integer Programming that considers value efficiency...")
    start_time = time.time()
    ip_result = optimizer.optimize_with_ip()
    ip_time = time.time() - start_time

    # Reset placement dictionary for the next optimization
    optimizer.placement = {}
    optimizer.memo = {}

    # Run and time the DP-based optimization
    print("\nRunning optimization with Dynamic Programming that considers value efficiency...")
    start_time = time.time()
    dp_result = optimizer.optimize_with_dp()
    dp_time = time.time() - start_time

    # Print comparison in a formatted table
    print("\n" + "="*90)
    print(f"{'COMPARISON OF EFFICIENCY-BASED OPTIMIZATION APPROACHES':^90}")
    print("="*90)

    headers = ["Metric", "IP-Based (Efficiency)", "DP-Based (Efficiency)"]
    print(f"{headers[0]:<30} {headers[1]:<30} {headers[2]:<30}")
    print("-"*90)

    metrics = [
        ["Total Cost (INR)", f"₹{ip_result['packer_cost']:.2f}", f"₹{dp_result['packer_cost']:.2f}"],
        ["Total Value (INR)", f"₹{ip_result['total_value']:.2f}", f"₹{dp_result['total_value']:.2f}"],
        ["Cabin Items", len(ip_result['cabin_items']), len(dp_result['cabin_items'])],
        ["Checkin Items", len(ip_result['checkin_items']), len(dp_result['checkin_items'])],
        ["Packer Items", len(ip_result['packer_items']), len(dp_result['packer_items'])],
        ["Cabin Weight (kg)", f"{ip_result['cabin_weight']:.2f}/{cabin_weight_limit}", f"{dp_result['cabin_weight']:.2f}/{cabin_weight_limit}"],
        ["Cabin Volume (L)", f"{ip_result['cabin_volume']:.2f}/{cabin_volume_limit}", f"{dp_result['cabin_volume']:.2f}/{cabin_volume_limit}"],
        ["Cabin Value/Weight (₹/kg)", f"{ip_result['cabin_value_per_weight']:.2f}", f"{dp_result['cabin_value_per_weight']:.2f}"],
        ["Cabin Value/Volume (₹/L)", f"{ip_result['cabin_value_per_volume']:.2f}", f"{dp_result['cabin_value_per_volume']:.2f}"],
        ["Checkin Weight (kg)", f"{ip_result['checkin_weight']:.2f}/{checkin_weight_limit}", f"{dp_result['checkin_weight']:.2f}/{checkin_weight_limit}"],
        ["Checkin Volume (L)", f"{ip_result['checkin_volume']:.2f}/{checkin_volume_limit}", f"{dp_result['checkin_volume']:.2f}/{checkin_volume_limit}"],
        ["Checkin Value/Weight (₹/kg)", f"{ip_result['checkin_value_per_weight']:.2f}", f"{dp_result['checkin_value_per_weight']:.2f}"],
        ["Checkin Value/Volume (₹/L)", f"{ip_result['checkin_value_per_volume']:.2f}", f"{dp_result['checkin_value_per_volume']:.2f}"],
        ["Computation Time (s)", f"{ip_time:.4f}", f"{dp_time:.4f}"]
    ]

    for metric in metrics:
        print(f"{metric[0]:<30} {metric[1]:<30} {metric[2]:<30}")

    print("-"*90)


    # Print detailed results for the IP-based optimization
    print("\nDetailed Results for IP-Based Optimization:")
    print("="*80)

    print(f"Total Value of All Items: ₹{ip_result['total_value']:.2f}")
    print(f"Total Packer Cost: ₹{ip_result['packer_cost']:.2f}")

    print(f"\nCabin Baggage (Weight: {ip_result['cabin_weight']:.2f}/{cabin_weight_limit} kg, " +
          f"Volume: {ip_result['cabin_volume']:.2f}/{cabin_volume_limit} L):")
    print(f"Total Value: ₹{ip_result['cabin_value']:.2f}")
    print("-"*80)
    print(f"{'Item Name':<40} {'Weight (kg)':<15} {'Volume (L)':<15} {'Value (₹)':<15}")
    print("-"*80)
    for item in ip_result['cabin_items']:
        print(f"{item['name']:<40} {item['weight']:<15.2f} {item['volume']:<15.2f} {item['value']:<15.2f}")

    print(f"\nCheck-in Baggage (Weight: {ip_result['checkin_weight']:.2f}/{checkin_weight_limit} kg, " +
          f"Volume: {ip_result['checkin_volume']:.2f}/{checkin_volume_limit} L):")
    print(f"Total Value: ₹{ip_result['checkin_value']:.2f}")
    print("-"*80)
    print(f"{'Item Name':<40} {'Weight (kg)':<15} {'Volume (L)':<15} {'Value (₹)':<15}")
    print("-"*80)
    for item in ip_result['checkin_items']:
        print(f"{item['name']:<40} {item['weight']:<15.2f} {item['volume']:<15.2f} {item['value']:<15.2f}")

    print(f"\nItems via Packers & Movers (Total Volume: {ip_result['packer_volume']:.2f} L):")
    print(f"Total Value: ₹{ip_result['packer_value']:.2f}")
    print("-"*80)
    print(f"{'Item Name':<40} {'Weight (kg)':<15} {'Volume (L)':<15} {'Value (₹)':<15} {'Cost (₹)':<15}")
    print("-"*80)
    for item in ip_result['packer_items']:
        item_cost = item['volume'] * packer_cost
        print(f"{item['name']:<40} {item['weight']:<15.2f} {item['volume']:<15.2f} {item['value']:<15.2f} {item_cost:<15.2f}")
    # Print detailed results for the DP-based optimization
    print("\nDetailed Results for DP-Based Optimization:")
    print("="*80)

    print(f"Total Value of All Items: ₹{dp_result['total_value']:.2f}")
    print(f"Total Packer Cost: ₹{dp_result['packer_cost']:.2f}")

    print(f"\nCabin Baggage (Weight: {dp_result['cabin_weight']:.2f}/{cabin_weight_limit} kg, " +
          f"Volume: {dp_result['cabin_volume']:.2f}/{cabin_volume_limit} L):")
    print(f"Total Value: ₹{dp_result['cabin_value']:.2f}")
    print("-"*80)
    print(f"{'Item Name':<40} {'Weight (kg)':<15} {'Volume (L)':<15} {'Value (₹)':<15}")
    print("-"*80)
    for item in dp_result['cabin_items']:
        print(f"{item['name']:<40} {item['weight']:<15.2f} {item['volume']:<15.2f} {item['value']:<15.2f}")

    print(f"\nCheck-in Baggage (Weight: {dp_result['checkin_weight']:.2f}/{checkin_weight_limit} kg, " +
          f"Volume: {dp_result['checkin_volume']:.2f}/{checkin_volume_limit} L):")
    print(f"Total Value: ₹{dp_result['checkin_value']:.2f}")
    print("-"*80)
    print(f"{'Item Name':<40} {'Weight (kg)':<15} {'Volume (L)':<15} {'Value (₹)':<15}")
    print("-"*80)
    for item in dp_result['checkin_items']:
        print(f"{item['name']:<40} {item['weight']:<15.2f} {item['volume']:<15.2f} {item['value']:<15.2f}")

    print(f"\nItems via Packers & Movers (Total Volume: {dp_result['packer_volume']:.2f} L):")
    print(f"Total Value: ₹{dp_result['packer_value']:.2f}")
    print("-"*80)
    print(f"{'Item Name':<40} {'Weight (kg)':<15} {'Volume (L)':<15} {'Value (₹)':<15} {'Cost (₹)':<15}")
    print("-"*80)
    for item in dp_result['packer_items']:
        item_cost = item['volume'] * packer_cost
        print(f"{item['name']:<40} {item['weight']:<15.2f} {item['volume']:<15.2f} {item['value']:<15.2f} {item_cost:<15.2f}")