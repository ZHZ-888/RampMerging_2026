import numpy as np

def get_depature_timels(simulation_time, flow_rate, seed=None):
    """
    Calculate a list of random departure times for vehicles based on the given flow rate.

    Parameters:
    flow_rate (float): The vehicle flow rate per hour.
    fixed_simulation_time (int): The fixed simulation time in seconds, default is 3600 seconds.
    seed (int): Optional seed for the random number generator to ensure reproducibility.

    Returns:
    list: A list of random departure times for each vehicle (in seconds), with unique times.
    """
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Calculate the total number of vehicles to be generated
    total_vehicles = int(flow_rate * (simulation_time / 3600))

    # Generate exponential distribution for the time intervals between vehicle departures
    intervals = np.random.exponential(scale=(3600 / flow_rate), size=total_vehicles)

    # Add small random perturbation to ensure uniqueness
    perturbation = np.random.uniform(0, 0.1, size=total_vehicles)  # Small perturbation
    intervals += perturbation

    # Calculate cumulative sum of the intervals to get the departure times
    departure_times = np.cumsum(intervals)

    # Round departure times to the nearest integer
    departure_times = np.round(departure_times).astype(int)

    # Ensure departure times do not exceed the fixed simulation time
    departure_times = departure_times[departure_times < simulation_time]

    # Ensure unique departure times by adding 1 second to duplicates
    unique_departure_times = []
    last_time = -1
    for time in departure_times:
        if time <= last_time:
            time = last_time + 1
        unique_departure_times.append(time)
        last_time = time
    return unique_departure_times


if __name__ == '__main__':
    # Example usage
    departure_times = get_depature_timels(flow_rate=750, simulation_time=1000, seed=1)
    print(departure_times)
    print(len(departure_times))
