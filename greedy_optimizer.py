"""Assignment 5: Greedy Solution Designer.

Three greedy algorithms for the instructor's GreedyOptimizer repository:
    1. Activity selection (maximize completed deliveries)
    2. Fractional knapsack (maximize priority within truck capacity)
    3. Interval partitioning (minimize drivers for all deliveries)

Run ``py -3.14 greedy_optimizer.py`` from your project directory after running
``py -3.14 delivery_generator.py`` to execute the tests and scenario benchmarks.
"""

import heapq
import json
import time
from pathlib import Path


# ============================================================================
# PART 1: PACKAGE PRIORITIZATION (Activity Selection)
# ============================================================================
def maximize_deliveries(time_windows):
    """Return delivery IDs for a maximum-size non-overlapping schedule.

    Input: list of dictionaries with 'delivery_id', 'start', and 'end'.
    Greedy choice: always take the next delivery with earliest finish time.
    Time: O(n log n) for sorting; space: O(n).
    """
    # Sorting creates a new list without modifying the caller's input.
    sorted_windows = sorted(time_windows, key=lambda item: item['end'])
    selected_ids = []
    last_end_time = float('-inf')

    for delivery in sorted_windows:
        # Back-to-back time windows are compatible.
        if delivery['start'] >= last_end_time:
            selected_ids.append(delivery['delivery_id'])
            last_end_time = delivery['end']

    return selected_ids


# ============================================================================
# PART 2: TRUCK LOADING (Fractional Knapsack)
# ============================================================================
def optimize_truck_load(packages, weight_limit):
    """Return the maximum load priority, total weight, and selected fractions.

    Input: packages with 'package_id', 'weight', and 'priority'.
    Output: {'total_priority': number, 'total_weight': number,
             'packages': [{'package_id': id, 'fraction': number}, ...]}.
    Greedy choice: take the most priority per pound first.
    Time: O(n log n) for sorting; space: O(n).
    """
    if weight_limit < 0:
        raise ValueError('Truck weight_limit must be nonnegative')

    for package in packages:
        if package['weight'] <= 0:
            raise ValueError('Package weight must be positive')
        if package['priority'] < 0:
            raise ValueError('Package priority must be nonnegative')

    sorted_packages = sorted(
        packages,
        key=lambda item: item['priority'] / item['weight'],
        reverse=True,
    )

    total_priority = 0.0
    total_weight = 0.0
    selected_packages = []

    for package in sorted_packages:
        remaining_capacity = weight_limit - total_weight
        if remaining_capacity <= 0:
            break
        if package['priority'] == 0:
            # Loading a zero-priority package does not improve the objective.
            continue

        weight_taken = min(package['weight'], remaining_capacity)
        fraction = weight_taken / package['weight']

        selected_packages.append({
            'package_id': package['package_id'],
            'fraction': fraction,
        })
        total_weight += weight_taken
        total_priority += package['priority'] * fraction

    return {
        'total_priority': total_priority,
        'total_weight': total_weight,
        'packages': selected_packages,
    }


# ============================================================================
# PART 3: DRIVER ASSIGNMENT (Interval Partitioning)
# ============================================================================
def minimize_drivers(deliveries):
    """Assign every delivery using the fewest drivers.

    Input: list of dictionaries with 'delivery_id', 'start', 'end'.
    Output: {'num_drivers': int, 'assignments': list of lists of delivery IDs}.
    Sort by start time; reuse the earliest-available driver if possible.
    A min-heap keeps the earliest driver availability at the front.
    Time: O(n log n); space: O(n).
    """
    sorted_deliveries = sorted(deliveries, key=lambda item: item['start'])
    # Each heap entry stores (driver_end_time, driver_index).
    available_drivers = []
    assignments = []

    for delivery in sorted_deliveries:
        start = delivery['start']
        end = delivery['end']

        if available_drivers and available_drivers[0][0] <= start:
            _, driver_index = heapq.heappop(available_drivers)
        else:
            driver_index = len(assignments)
            assignments.append([])

        assignments[driver_index].append(delivery['delivery_id'])
        heapq.heappush(available_drivers, (end, driver_index))

    return {
        'num_drivers': len(assignments),
        'assignments': assignments,
    }


# ============================================================================
# TESTING & BENCHMARKING
# ============================================================================
def load_scenario(filename):
    """Read scenario JSON from the scenarios folder beside this Python file."""
    path = Path(__file__).resolve().parent / 'scenarios' / filename
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def test_package_prioritization():
    """Check activity selection on known cases."""
    print('\nTESTING PACKAGE PRIORITIZATION')
    cases = [
        ([
            {'delivery_id': 'A', 'start': 1, 'end': 2},
            {'delivery_id': 'B', 'start': 3, 'end': 4},
            {'delivery_id': 'C', 'start': 5, 'end': 6},
        ], 3),
        ([
            {'delivery_id': 'A', 'start': 1, 'end': 5},
            {'delivery_id': 'B', 'start': 2, 'end': 4},
            {'delivery_id': 'C', 'start': 3, 'end': 6},
        ], 1),
        ([
            {'delivery_id': 'A', 'start': 1, 'end': 3},
            {'delivery_id': 'B', 'start': 2, 'end': 5},
            {'delivery_id': 'C', 'start': 4, 'end': 7},
            {'delivery_id': 'D', 'start': 6, 'end': 9},
        ], 2),
        ([], 0),
    ]
    for number, (windows, expected) in enumerate(cases, 1):
        result = maximize_deliveries(windows)
        assert len(result) == expected, f'Test {number}: {result}'
        by_id = {item['delivery_id']: item for item in windows}
        for previous, current in zip(result, result[1:]):
            assert by_id[previous]['end'] <= by_id[current]['start']
        print(f'  Test {number}: PASS ({len(result)} deliveries)')


def test_truck_loading():
    """Check fractional knapsack on known cases."""
    print('\nTESTING TRUCK LOADING')
    packages = [
        {'package_id': 'A', 'weight': 10, 'priority': 60},
        {'package_id': 'B', 'weight': 20, 'priority': 100},
        {'package_id': 'C', 'weight': 30, 'priority': 120},
    ]
    all_fit = optimize_truck_load(packages[:2], 50)
    assert all_fit['total_priority'] == 160
    assert all_fit['total_weight'] == 30
    print('  Test 1: PASS (priority 160, weight 30)')

    partial = optimize_truck_load(packages, 50)
    assert abs(partial['total_priority'] - 240) < 1e-9
    assert abs(partial['total_weight'] - 50) < 1e-9
    assert len(partial['packages']) == 3
    assert abs(partial['packages'][-1]['fraction'] - (2 / 3)) < 1e-9
    print('  Test 2: PASS (priority 240, includes 2/3 of C)')

    empty = optimize_truck_load(packages, 0)
    assert empty['total_priority'] == 0 and empty['packages'] == []
    print('  Test 3: PASS (zero capacity)')


def test_driver_assignment():
    """Check interval partitioning on known cases."""
    print('\nTESTING DRIVER ASSIGNMENT')
    cases = [
        ([
            {'delivery_id': 'A', 'start': 1, 'end': 2},
            {'delivery_id': 'B', 'start': 3, 'end': 4},
            {'delivery_id': 'C', 'start': 5, 'end': 6},
        ], 1),
        ([
            {'delivery_id': 'A', 'start': 1, 'end': 5},
            {'delivery_id': 'B', 'start': 2, 'end': 6},
            {'delivery_id': 'C', 'start': 3, 'end': 7},
        ], 3),
        ([
            {'delivery_id': 'A', 'start': 1, 'end': 3},
            {'delivery_id': 'B', 'start': 2, 'end': 4},
            {'delivery_id': 'C', 'start': 4, 'end': 6},
            {'delivery_id': 'D', 'start': 5, 'end': 7},
        ], 2),
        ([], 0),
    ]
    for number, (deliveries, expected) in enumerate(cases, 1):
        result = minimize_drivers(deliveries)
        assert result['num_drivers'] == expected, f'Test {number}: {result}'
        actual_ids = [i for driver in result['assignments'] for i in driver]
        assert sorted(actual_ids) == sorted(i['delivery_id'] for i in deliveries)
        by_id = {item['delivery_id']: item for item in deliveries}
        for driver in result['assignments']:
            for first, second in zip(driver, driver[1:]):
                assert by_id[first]['end'] <= by_id[second]['start']
        print(f'  Test {number}: PASS ({result["num_drivers"]} drivers)')


def benchmark_scenarios():
    """Measure performance on the generated assignment scenarios."""
    print('\nBENCHMARKING ON REALISTIC SCENARIOS')

    windows = load_scenario('package_prioritization.json')
    start = time.perf_counter()
    selected = maximize_deliveries(windows)
    elapsed = (time.perf_counter() - start) * 1000
    print(f'  Package prioritization: {len(selected)}/{len(windows)} deliveries, '
          f'{elapsed:.4f} ms')

    data = load_scenario('truck_loading.json')
    start = time.perf_counter()
    load = optimize_truck_load(data['packages'], data['truck_capacity'])
    elapsed = (time.perf_counter() - start) * 1000
    print(f'  Truck loading: priority {load["total_priority"]:.2f}, '
          f'weight {load["total_weight"]:.2f} lbs, '
          f'{len(load["packages"])} packages, {elapsed:.4f} ms')

    deliveries = load_scenario('driver_assignment.json')
    start = time.perf_counter()
    drivers = minimize_drivers(deliveries)
    elapsed = (time.perf_counter() - start) * 1000
    print(f'  Driver assignment: {drivers["num_drivers"]} drivers for '
          f'{len(deliveries)} deliveries, {elapsed:.4f} ms')


if __name__ == '__main__':
    print('ASSIGNMENT 5: GREEDY SOLUTION DESIGNER')
    test_package_prioritization()
    test_truck_loading()
    test_driver_assignment()

    scenarios_dir = Path(__file__).resolve().parent / 'scenarios'
    expected_files = (
        'package_prioritization.json',
        'truck_loading.json',
        'driver_assignment.json',
    )
    if all((scenarios_dir / name).is_file() for name in expected_files):
        benchmark_scenarios()
    else:
        print('\nScenario JSON files not found. Run delivery_generator.py '
              'in the same project folder, then run this file again to benchmark.')

    print('\nALL SMALL TESTS PASSED')
