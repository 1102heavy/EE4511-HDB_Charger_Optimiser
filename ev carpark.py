import simpy
import random
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.express as px
import datetime
# --- 1. Input Parameters ---
# (Replace placeholder values with actual data for the specific HDB block)

# Simulation Parameters
SIM_DURATION_DAYS = 30
SIM_DURATION_MINUTES = SIM_DURATION_DAYS * 24 * 60
RANDOM_SEED = 42 # For reproducible results

# HDB Car Park & EV Parameters
TOTAL_PARKING_LOTS = 430
ESTIMATED_CAR_OWNERSHIP_RATE = 0.7 # User specified
ESTIMATED_INITIAL_EV_PENETRATION = 0.5 # User specified
EV_PENETRATION_GROWTH_RATE = 0.1 # Annual growth rate assumption
TARGET_YEAR = 5 # Optimization target year

# Calculate number of EVs for the target year
total_cars = TOTAL_PARKING_LOTS * ESTIMATED_CAR_OWNERSHIP_RATE
projected_ev_penetration = ESTIMATED_INITIAL_EV_PENETRATION * ( (1 + EV_PENETRATION_GROWTH_RATE) ** TARGET_YEAR )
NUM_EVS = int(total_cars * projected_ev_penetration)
print(f"Simulating with {NUM_EVS} EVs for target year {TARGET_YEAR}.")

# EV Characteristics (Average values, consider distributions for more realism)
AVG_BATTERY_CAPACITY_KWH = 60.0
MIN_SOC_THRESHOLD = 0.2 # Arrive when SOC drops below this
MAX_SOC_TARGET = 0.8 # Target SOC after charging
AVG_ENERGY_CONSUMPTION_KWH_PER_KM = 0.18
AVG_DAILY_MILEAGE_KM = 50

# Charging Behavior Assumptions
CHARGE_FREQUENCY_DAYS_MEAN = 5 # Avg days between needing a charge (not used anymore)
CHARGE_FREQUENCY_DAYS_STDDEV = 1.5 # Not used anymore

# --- *** MODIFIED: Only DC_150kw chargers available *** ---
# Charger Types and Costs (Unit + Installation)
CHARGER_TYPES = {
    'DC_150kw': {'power': 150.0, 'cost_unit': 35000, 'cost_install': 4500},
}
# Fixed order for optimization variables (array 'x' in objective_function)
CHARGER_TYPE_ORDER = list(CHARGER_TYPES.keys()) # Now just ['DC_150kw']

# Infrastructure Costs Assumptions
BASE_INFRA_COST = 3200 # Fixed cost for grid connection, etc.
INFRA_COST_PER_KW = 100 # Variable cost based on total installed power

# --- Optimization Parameters ---
MAX_P95_WAITING_TIME_MINUTES = 45 # Constraint: 95% of waits <= this value
MAX_BUDGET = 250000 # Constraint: Total installation cost <= this value

# --- 2. Simulation Components ---
# Suppose we define a 'grid_capacity_kW' of 500
#GRID_CAPACITY_KW = 500 # Total grid capacity in kW

# Change
class GridCapacity:
    """A simple resource that tracks how many kW are currently in use."""
    def __init__(self, env, total_kw):
        self.env = env
        self._capacity = simpy.Container(env, capacity=total_kw, init=total_kw)

    def request_power(self, amount_kw):
        """Request a certain kW from the grid capacity."""
        return self._capacity.get(amount_kw)

    def release_power(self, amount_kw):
        """Release kW back to the grid."""
        return self._capacity.put(amount_kw)
#change


class ChargingStation:
    """ Represents the charging station with multiple chargers managed by SimPy Resources. """
    def __init__(self, env, charger_config, grid_capacity):
        self.env = env
        self.grid_capacity = grid_capacity
        self.chargers = {}  # Stores simpy.Resource objects for each charger type
        self.waiting_times = []
        self.total_energy_delivered = 0.0
        self.successful_charges = 0
        self.failed_charges = 0
        self.dynamic_cost_spent = 0.0

        # Create SimPy resources based on the provided configuration
        for type_name, count in charger_config.items():
            count = max(0, int(round(count)))  # Ensure non-negative integer
            if count > 0 and type_name in CHARGER_TYPES:
                self.chargers[type_name] = simpy.Resource(env, capacity=count)

    def record_waiting_time(self, wait):
        self.waiting_times.append(wait)
    def get_charger_resource(self, charger_type):
        """ Returns the SimPy resource for the requested charger type, if available. """
        return self.chargers.get(charger_type)

    def record_energy(self, energy):
        self.total_energy_delivered += energy
        self.successful_charges += 1

    def record_failed_charge(self):
        self.failed_charges += 1
        print(f"[{self.env.now}] Failed charges incremented to {self.failed_charges}")

    # ---------------- NEW VERSION OF THIS METHOD ----------------
    def add_one_dc_charger(self):
        """
        Creates a new simpy.Resource for 'DC_150kw' with capacity = old_capacity + 1,
        replaces the existing one in self.chargers,
        and adds the cost to self.dynamic_cost_spent.
        """
        old_res = self.chargers.get('DC_150kw')
        if not old_res:
            # If there's no existing DC_150kw resource, we can't "expand" it.
            return

        old_capacity = old_res.capacity
        new_capacity = old_capacity + 1

        # Create a brand-new Resource with the new capacity
        new_res = simpy.Resource(self.env, capacity=new_capacity)

        # Replace the old resource in our dictionary
        self.chargers['DC_150kw'] = new_res

        # Add cost for installing one more DC_150kw charger
        new_charger_cost = (
            CHARGER_TYPES['DC_150kw']['cost_unit']
            + CHARGER_TYPES['DC_150kw']['cost_install']
        )
        self.dynamic_cost_spent += new_charger_cost

        print(
            f"[{self.env.now:.1f}] Replaced DC_150kw resource: "
            f"old capacity={old_capacity}, new capacity={new_capacity}, "
            f"cost={new_charger_cost:.1f}"
        )
    def expand_grid(self, amount):
        old_container = self.grid_capacity._capacity
        old_cap = old_container.capacity
        new_capacity = old_cap + amount

        new_container = simpy.Container(
            self.env,
            capacity=new_capacity,
            init=new_capacity  # fully available
        )
        self.grid_capacity._capacity = new_container

        cost_grid = amount * 50.0
        self.dynamic_cost_spent += cost_grid

        print(f"[{self.env.now:.1f}] Grid expanded by {amount} kW, cost={cost_grid:.1f}. "
            f"New Container capacity={new_capacity}")




def ev(name, env, station, battery_kwh, min_soc, max_soc):
    """
    An example EV process with weekday vs. weekend logic
    and an optional midday "lunch charge".
    """
    # Initialize the EV's SOC randomly between 80-100% (just an example)
    current_soc = random.uniform(0.8, 1.0)

    # We'll simulate day-by-day
    while True:
        current_day = int(env.now // (24 * 60))  # how many days have elapsed
        day_of_week = current_day % 7           # 0=Monday, 6=Sunday

        # --- 1. Decide how much the EV drives today ---
        if day_of_week < 5:
            # Weekday driving
            daily_km = random.gauss(50, 10)     # e.g. 50 +/- 10 km
        else:
            # Weekend driving
            daily_km = random.gauss(10, 5)      # e.g. 10 +/- 5 km

        # Convert daily km to daily kWh usage
        # You might keep a global average consumption (e.g. 0.18 kWh/km)
        # or pass it in as a parameter. We'll define it inline here:
        kwh_per_km = 0.18
        daily_kwh_used = max(0, daily_km * kwh_per_km)

        soc_drop_for_day = daily_kwh_used / battery_kwh
        # The EV won't "use" it instantly; let's break the day into segments.

        # --- 2. MORNING COMMUTE (?) ---
        # We'll assume the EV does some morning usage
        morning_commute_kwh = soc_drop_for_day * 0.5  # Half usage in morning
        yield env.timeout(60)  # e.g. 1 hour to represent morning usage
        current_soc = max(0, current_soc - morning_commute_kwh)

        # --- 3. MIDDAY CHARGE (Optional "lunch time" if at work) ---
        # e.g. 50% chance they have access to a charger at midday
        if random.random() < 0.5:
            # Possibly do a midday top-up if min_soc is reached
            if current_soc < min_soc:
                yield from do_charge(env, station, 'DC_150kw', current_soc, battery_kwh, min_soc, max_soc, name)
        # Simulate the midday hours
        yield env.timeout(3 * 60)  # e.g. 3 hours lunch/work

        # --- 4. AFTERNOON USAGE ---
        afternoon_commute_kwh = soc_drop_for_day * 0.5
        yield env.timeout(60)  # another hour
        current_soc = max(0, current_soc - afternoon_commute_kwh)

        # --- 5. EVENING CHARGE (Home) ---
        if current_soc < min_soc:
            # e.g. Attempt a full charge at home
            yield from do_charge(env, station, 'DC_150kw', current_soc, battery_kwh, min_soc, max_soc, name)

        # Finally, pass the night hours
        # or you can do multiple small time steps if you prefer
        yield env.timeout(24*60 - (5*60))  
        # (We used 1+3+1=5 hours above, so the rest of 24h is 19 hours left.)

        # Possibly move to the next day. The loop continues.

def do_charge(env, station, charger_type, current_soc, battery_kwh, min_soc, max_soc, ev_name):
    """
    Attempt to do a charge on the specified charger type.
    Returns a generator that can be 'yield from' in the main EV loop.
    """
    arrival_time = env.now
    charger_resource = station.get_charger_resource(charger_type)

    if not charger_resource:
        station.record_failed_charge()
        return

    needed_kw = CHARGER_TYPES[charger_type]['power']

    try:
        with charger_resource.request() as req:
            yield req

            grid_req = station.grid_capacity.request_power(needed_kw)
            # Wait up to 30 minutes for capacity
            results = yield grid_req | env.timeout(45)
            if grid_req not in results:
                print(f"{ev_name} can't get grid power in 30 min at {env.now}. Leaves.")
                station.record_failed_charge()
                return

            # Now do the actual charge calc
            kwh_to_charge = (max_soc - current_soc) * battery_kwh
            if kwh_to_charge <= 0.01:
                return  # negligible need

            charge_duration_minutes = (kwh_to_charge / needed_kw) * 60
            yield env.timeout(max(0.1, charge_duration_minutes))

            station.record_waiting_time(env.now - arrival_time)
            station.record_energy(kwh_to_charge)

    except simpy.Interrupt:
        print(f"WARN: {ev_name} charging interrupted at {env.now:.2f}")
        station.record_failed_charge()
    except Exception as e:
        print(f"ERROR for {ev_name} at {env.now:.2f}: {e}")
        station.record_failed_charge()
    finally:
        try:
            if env.now < SIM_DURATION_MINUTES:
                yield station.grid_capacity.release_power(needed_kw)
        except GeneratorExit:
            pass

    # If we reach here, we assume final SOC = max_soc


def setup_simulation(env, num_evs, charger_config, grid_capacity_kW):
        # 1) Create a GridCapacity object
    grid_cap = GridCapacity(env, grid_capacity_kW)

    """ Creates the ChargingStation and EV processes for the simulation. """
    station = ChargingStation(env, charger_config, grid_cap)

    # Create EV processes only if num_evs > 0
    if num_evs > 0:
        for i in range(num_evs):
            # Assign slightly randomized parameters to each EV
            daily_km = random.gauss(AVG_DAILY_MILEAGE_KM, 15)
            batt_kwh = random.gauss(AVG_BATTERY_CAPACITY_KWH, 10)
            batt_kwh = max(10, batt_kwh) # Ensure minimum battery size

            env.process(ev(f'EV_{i}', env, station,
                           battery_kwh=batt_kwh,
                           min_soc=MIN_SOC_THRESHOLD,
                           max_soc=MAX_SOC_TARGET))
            
            # Stagger EV starts slightly to avoid simultaneous initial checks
            if i < num_evs - 1:
                 yield env.timeout(random.uniform(0, 1))
    else:
         yield env.timeout(0) # Allow setup to complete if no EVs

    # Create a list to store (time, usage)
    usage_log = []

    # Start the monitor process
    monitoring_proc = env.process(monitor_grid_usage(env, grid_cap, interval=10, usage_log=usage_log))
    # e.g. sample every 10 minutes, adjust as desired
    # Start expansions manager
    env.process(expansions_manager(env, station))
    return station , usage_log # Return the station instance


# --- 3. Cost Calculation Function ---

def calculate_total_cost(charger_config):
    """ Calculates the total installation cost (CAPEX) for a given configuration. """
    total_cost = BASE_INFRA_COST
    total_power_kw = 0

    # Ensure counts are non-negative integers
    clean_config = {k: max(0, int(round(v))) for k, v in charger_config.items()}

    for type_name, count in clean_config.items():
        if count > 0 and type_name in CHARGER_TYPES:
            type_info = CHARGER_TYPES[type_name]
            # Cost = Count * (Unit Cost + Install Cost)
            total_cost += count * (type_info['cost_unit'] + type_info['cost_install'])
            total_power_kw += count * type_info['power']
        elif count > 0: # Check if type_name exists but wasn't in CHARGER_TYPES
             print(f"Warning: Charger type '{type_name}' in config but not defined in CHARGER_TYPES.")

    # Add infrastructure cost component based on total installed power
    total_cost += total_power_kw * INFRA_COST_PER_KW

    return total_cost

# --- 4. Optimization Framework ---

def run_simulation_and_get_metrics(charger_config, grid_capacity_kW):
    """
    Runs a simulation for a given charger config and grid capacity,
    and returns (metrics, station_instance).
    
    'metrics' is a dictionary with performance stats.
    'station_instance' is the ChargingStation object at the end,
    which can be used to inspect dynamic_cost_spent, expansions, etc.
    """

    # 1) Clean up the charger_config (round & ensure non-negative integer)
    sim_config = {k: max(0, int(round(v))) for k, v in charger_config.items()}

    # 2) If no chargers but EVs > 0 => immediate worst-case
    if sum(sim_config.values()) == 0 and NUM_EVS > 0:
        # Return infinite waiting etc.
        metrics = {
            'avg_wait_time': float('inf'), 'max_wait_time': float('inf'),
            'p95_wait_time': float('inf'),
            'total_energy': 0,
            'successful_charges': 0,
            'failed_charges': NUM_EVS * SIM_DURATION_DAYS,
            'usage_log': []
        }
        # Return a dummy station_instance or None
        return metrics, None

    # 3) Create & run the simulation
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    
    station_instance = None
    usage_log = []

    try:
        # 4) Start the setup process (which might create expansions_manager, if you have one)
        setup_proc = env.process(
            setup_simulation(env, NUM_EVS, sim_config, grid_capacity_kW)
        )
        env.run(until=setup_proc)
        
        # The value returned from setup_proc is the station_instance + usage_log
        station_instance, usage_log = setup_proc.value
        
        # Check we got a valid ChargingStation
        if not isinstance(station_instance, ChargingStation):
            raise RuntimeError("setup_simulation failed to return a ChargingStation object")

        # 5) Run the main simulation phase until SIM_DURATION_MINUTES
        env.run(until=SIM_DURATION_MINUTES)

    except Exception as e:
        print(f"Error during simulation run for config {sim_config}: {e}")
        # On error, return worst-case metrics + zero station
        metrics = {
            'avg_wait_time': float('inf'), 'max_wait_time': float('inf'),
            'p95_wait_time': float('inf'),
            'total_energy': 0,
            'successful_charges': 0,
            'failed_charges': NUM_EVS,
            'usage_log': []
        }
        return metrics, station_instance

    # 6) Compute final performance metrics from station_instance
    if station_instance and station_instance.waiting_times:
        avg_wait = np.mean(station_instance.waiting_times)
        max_wait = np.max(station_instance.waiting_times)
        p95_wait = np.percentile(station_instance.waiting_times, 95)
    else:
        avg_wait, max_wait, p95_wait = 0, 0, 0

    # 7) Build the metrics dictionary
    metrics = {
        'avg_wait_time': avg_wait,
        'max_wait_time': max_wait,
        'p95_wait_time': p95_wait,
        'total_energy': station_instance.total_energy_delivered if station_instance else 0,
        'successful_charges': station_instance.successful_charges if station_instance else 0,
        'failed_charges': station_instance.failed_charges if station_instance else NUM_EVS,
        'usage_log': usage_log  # attach usage log
    }

    # 8) Return both the final metrics and the final ChargingStation
    return metrics, station_instance


def objective_function(x):
    """
    Modified objective function for a 2D optimization of:
      x[0] = initial number of DC_150kw chargers
      x[1] = initial grid capacity (kW)
    
    We incorporate:
      - Up front cost for initial chargers + grid capacity
      - Dynamic expansions cost mid-run (recorded in station.dynamic_cost_spent)
      - A huge penalty for any failed charges
      - P95 wait time penalty if it exceeds MAX_P95_WAITING_TIME_MINUTES
      - Budget penalty if total cost exceeds MAX_BUDGET
    """

    # 1) Parse out the two decision variables
    init_num_chargers = max(0, int(round(x[0])))
    init_grid_capacity = max(0, float(x[1]))

    # 2) Build a small config dictionary for initial station
    charger_config = {'DC_150kw': init_num_chargers}

    # 3) Calculate the up‐front cost for these initial chargers
    base_charger_cost = calculate_total_cost(charger_config)

    # 4) Suppose we pay $50/kW for the initial grid capacity
    base_grid_cost = init_grid_capacity * 50.0

    # 5) So the base up‐front cost is:
    base_cost = base_charger_cost + base_grid_cost

    # 6) Run the simulation with these initial values. 
    #    The station can expand mid-run if you have expansions_manager, 
    #    so additional dynamic_cost_spent may accumulate.
    metrics, station_instance =run_simulation_and_get_metrics(
        charger_config, 
        init_grid_capacity
    )

       # 5) If the simulation returned None for station_instance, then we know it failed
    if station_instance is None:
        # The simulation had an error or crashed, so assign a large penalty
        return 1e9  # or something else suitably large
    # 7) Get the expansions cost that was accrued mid-run
    #    (You must store expansions in station_instance.dynamic_cost_spent)
    expansions_cost = getattr(station_instance, 'dynamic_cost_spent', 0.0)

    # 8) The total cost so far is base up‐front + expansions
    cost_so_far = base_cost + expansions_cost

    # 9) Evaluate performance and add penalties

    # a) If any EV fails (car leaves), that’s a big penalty 
    #    to ensure the solver invests enough to avoid fails.
    fail_count = station_instance.failed_charges
    fail_penalty = 0.0
    if fail_count > 0:
        fail_penalty = fail_count * 1e7  # or any large number

    # b) p95 wait time penalty
    p95_wait = metrics.get('p95_wait_time', float('inf'))
    wait_penalty = 0.0
    if p95_wait > MAX_P95_WAITING_TIME_MINUTES:
        wait_penalty = (p95_wait - MAX_P95_WAITING_TIME_MINUTES) * 5000

    # c) Budget penalty
    budget_penalty = 0.0
    if cost_so_far > MAX_BUDGET:
        budget_penalty = (cost_so_far - MAX_BUDGET) * 100.0

    # 10) Final objective = cost + all penalties
    final_obj = cost_so_far + fail_penalty + wait_penalty + budget_penalty

    return final_obj


def monitor_grid_usage(env, grid_capacity, interval, usage_log):
    """
    Monitors the grid usage every 'interval' simulation minutes.
    usage_log is a list where we'll store (time, kW_in_use).
    """
    while True:
        yield env.timeout(interval)
        current_level = grid_capacity._capacity.level
        total_capacity = grid_capacity._capacity.capacity  # read the container's max capacity
        kW_in_use = total_capacity - current_level
        usage_log.append((env.now, kW_in_use))

def expansions_manager(env, station):
    """
    Runs in parallel, checks if expansions are needed every X minutes.
    If queue is too long or usage is too high for a sustained time,
    invests in expansions mid-run.
    Also, if station.failed_charges has increased, we add new charger(s).
    """
    check_interval = 10.0  # minutes
    queue_threshold = 5
    queue_high_since = None
    
    usage_threshold_ratio = 0.9
    usage_high_since = None
    
    sustain_minutes = 60.0  # how long a condition must hold before we expand
    
    # *** NEW: track how many failed charges so far, so we can detect an increase
    last_failed_charges = 0
    
    while True:
        yield env.timeout(check_interval)
        
        # 1) Check DC queue length
        dc_resource = station.chargers.get('DC_150kw')
        if dc_resource:
            qlen = len(dc_resource.queue)
            if qlen >= queue_threshold:
                if queue_high_since is None:
                    queue_high_since = env.now
                else:
                    # Has it been high for a 'sustain_minutes' period?
                    if (env.now - queue_high_since) >= sustain_minutes:
                        station.add_one_dc_charger()
                        queue_high_since = None
            else:
                queue_high_since = None
        
        # 2) Check grid usage ratio
        used = station.grid_capacity._capacity.capacity - station.grid_capacity._capacity.level
        cap = station.grid_capacity._capacity.capacity
        ratio = used / cap if cap > 0 else 0
        if ratio >= usage_threshold_ratio:
            if usage_high_since is None:
                usage_high_since = env.now
            else:
                if (env.now - usage_high_since) >= sustain_minutes:
                    station.expand_grid(100.0)  # or whatever expansion step
                    usage_high_since = None
        else:
            usage_high_since = None
        
        # 3) *** NEW: Check if 'failed_charges' has increased
        current_failed = station.failed_charges
        # e.g. for every 1 new failed charge, add a new charger
        # or for every 2-3 new fails, add 1 charger, etc.
        new_fails = current_failed - last_failed_charges
        if new_fails > 0:
            # For each new failure, we add a DC charger
            # or maybe add one charger if new_fails >= some threshold, etc.
            for i in range(new_fails):
                station.add_one_dc_charger()
            
            # Update the memory of how many we've seen
            last_failed_charges = current_failed

# --- 5. Main Execution Block ---

# --- 5. Main Execution Block ---
if __name__ == '__main__':

    ######################################################
    # 1) MANUAL EXAMPLE (TWO PARAMETERS)
    ######################################################
    print("--- Running Manual Simulation Example ---")

    # Suppose we try 20 chargers, 600 kW
    manual_num_chargers = 20
    manual_grid_kW = 600

    # Build the config dictionary for chargers
    test_config = {'DC_150kw': manual_num_chargers}

    # 1) Compute base cost from chargers
    test_cost_chargers = calculate_total_cost(test_config)

    # 2) Add cost for the grid capacity, e.g. $50/kW
    cost_for_grid = manual_grid_kW * 50.0
    total_cost = test_cost_chargers + cost_for_grid

    print(f"Manual Example: {manual_num_chargers} chargers, {manual_grid_kW} kW grid")
    print(f"Charger cost: ${test_cost_chargers:.2f}, Grid cost: ${cost_for_grid:.2f}, Total: ${total_cost:.2f}")

    if total_cost <= MAX_BUDGET:
        # If within budget, we can run the simulation
        metrics, station_obj = run_simulation_and_get_metrics(test_config, manual_grid_kW)
        print(f"Simulation Metrics: {metrics}")
    else:
        # If over budget, either skip or still run
        print("Manual example is over budget; skipping or do something else.")
        metrics = {}

    print("-" * 30)


    ######################################################
    # 2) OPTIMIZATION RUN (TWO VARIABLES)
    ######################################################
    print("\n--- Optimization Run (DC_150kw + Grid Capacity) ---")

    # For example:
    # x[0] in [0..40]  => # of DC chargers
    # x[1] in [100..1000] => grid capacity (kW)
    bounds = [(0, 40), (100, 1000)]
    print(f"Bounds: {bounds}")
    print("Starting optimization with differential_evolution (2D)...")
    print(f"Performance constraint: p95 wait <= {MAX_P95_WAITING_TIME_MINUTES} min")
    print(f"Budget constraint: <= ${MAX_BUDGET:.2f}")

    try:
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            strategy='best1bin',
            maxiter=50,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=RANDOM_SEED,
            disp=True,
            polish=True,
            # If you want both integer, you can do integrality=[True, True]
            integrality=[True, False]
        )

        if result.success or 'maximum number of generations' in result.message.lower():
            # Decode the best solution
            optimal_num_chargers = max(0, int(round(result.x[0])))
            optimal_grid_kW = max(0, int(round(result.x[1])))

            # Build the config for that many chargers
            optimal_config = {'DC_150kw': optimal_num_chargers}

            # Calculate cost (charger + grid)
            cost_chargers = calculate_total_cost(optimal_config)
            cost_grid = optimal_grid_kW * 50.0  # or your actual cost formula
            optimal_cost = cost_chargers + cost_grid

            print("\n--- Optimization Result ---")
            print(f"Termination Message: {result.message}")
            print(f"Optimal # DC chargers: {optimal_num_chargers}")
            print(f"Optimal Grid Capacity: {optimal_grid_kW} kW")
            print(f"Objective Function Value: {result.fun:.2f}")
            print(f"Recalculated Cost: ${optimal_cost:.2f}")

            # Rerun the simulation to get final metrics:
            print("\n--- Running Simulation with Optimal Config ---")
            final_metrics, station_obj = run_simulation_and_get_metrics(optimal_config, optimal_grid_kW)
            print(f"Metrics for Optimal Config: {final_metrics}")

            # Now station_obj is defined. Let's see how many chargers and final grid capacity exist:
            final_dc_res = station_obj.chargers.get('DC_150kw', None)
            if final_dc_res:
                print("Final DC Charger capacity:", final_dc_res.capacity)
            else:
                print("No DC_150kw chargers in station.")

            # Check final grid capacity
            # (Expansions during the simulation might have altered it)
            final_grid_capacity = station_obj.grid_capacity._capacity.capacity
            print("Final Grid capacity (kW):", final_grid_capacity)

            # Check budget
            final_budget_check = MAX_BUDGET - optimal_cost
            print(f"Budget Check: {final_budget_check:.2f} (>=0 is good)")
            if final_budget_check < 0:
                print("Warning: solution might exceed budget slightly.")

            # Finally, check p95 wait
            final_p95_wait_check = final_metrics.get('p95_wait_time', float('inf'))
            print(f"P95 Wait Time: {final_p95_wait_check:.2f} (<= {MAX_P95_WAITING_TIME_MINUTES} min?)")
            if final_p95_wait_check > MAX_P95_WAITING_TIME_MINUTES:
                print("Warning: solution's P95 wait might exceed constraint.")
        else:
            # If optimization didn't converge
            print("\n--- Optimization Failed or Did Not Converge ---")
            print(f"Status: {result.message}")
            print(f"Final Iteration Variables: {result.x}")
            print(f"Final Objective Value: {result.fun}")

    except Exception as e:
        print("\n--- An error occurred during optimization ---")
        print(e)


    ######################################################
    # 3) PLOTTING THE USAGE DATA
    ######################################################
    usage_data = metrics.get('usage_log', [])
    if usage_data:
        times_in_minutes = [pt[0] for pt in usage_data]
        usage = [pt[1] for pt in usage_data]

        start_time = pd.Timestamp("2023-01-01 00:00:00")
        times_dt = [start_time + pd.Timedelta(minutes=m) for m in times_in_minutes]

        fig = px.line(
            x=times_dt,
            y=usage,
            labels={"x": "Time", "y": "Grid kW in use"}
        )

        # tickformatstops example
        fig.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 1000], value="%H:%M:%S.%L ms"),
                dict(dtickrange=[1000, 60000], value="%H:%M:%S"),
                dict(dtickrange=[60000, 3600000], value="%H:%M"),
                dict(dtickrange=[3600000, 86400000], value="%d %H:%M"),
                dict(dtickrange=[86400000, None], value="%d %b")
            ]
        )

        fig.update_layout(title="Grid Power Usage Over 30 Days, Zoom for Hours/Seconds")
        fig.show()
    else:
        print("No usage data available to plot.")

    print("\n--- End of Script ---")
    print("Remember to replace placeholder data and refine logic.")
    print("Consider using multiple simulation runs for robustness.")
    print("You might need to adjust cost formulas, integrality, etc.")



