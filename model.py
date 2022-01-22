'''
RoboMarkt project - Foundations of Operations Reaserch
Students devs: Ottavia Belotti, Alessio Braccini, Martin Bronzo
'''

from math import sqrt, ceil
from sys import stdout as out
from sys import argv
from mip import Model, xsum, minimize, BINARY
import bisect


# CLARKE-WRIGHT ALGORITHM
def get_savings(built_stores, distance_matrix):
    savings = []
    saving_list = []
    for store_count, i in enumerate(built_stores[1:]):
        for j in built_stores[store_count + 2:]:
            saving = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            line = [i, j, saving]
            k = bisect.bisect(saving_list, saving)
            saving_list.insert(k, saving)
            savings.insert(len(savings) - k, line)
    return savings


def finalize_remaining_shops_routes(stores):
    routes = []

    n_nodes = 0
    n_nodes_in_route = 0
    available_stores_ind = stores.copy()
    prev = 0

    print("Available stores: " + str(available_stores_ind))
    while n_nodes < len(stores):
        available_distance = list(map(lambda d: distance[prev][d], available_stores_ind))

        # choose the closest available store as next
        next_list = list(filter(lambda available_store: distance[prev][available_store] == min(available_distance),
                                available_stores_ind))
        if len(next_list) > 0:
            next_s = next_list[0]
        else:
            raise ValueError from None
        # remove chosen store from the available ones
        available_stores_ind.remove(next_s)
        # add chosen store in route
        routes.append(next_s)
        n_nodes += 1
        n_nodes_in_route += 1
        # now next has been visited so make it the starting point
        prev = next_s

    return routes


def get_routes(stores, savings, capacity, n_trucks):
    route = [savings[0][0], savings[0][1]]
    stores_to_be_added = stores[1:]
    stores_to_be_added.remove(savings[0][0])
    stores_to_be_added.remove(savings[0][1])
    routes = [route]
    savings.pop(0)
    for saving in savings:
        added_just_one_constraint = 0
        added_both_different_routes_constraint = 0
        added_both_same_route_constraint = 0
        first_external = 0
        second_external = 0
        first_route_index = -1
        second_route_index = -1
        for index, route in enumerate(routes):
            x = route.count(saving[0])
            y = route.count(saving[1])
            # are both in the same route
            if x and y:
                added_both_same_route_constraint = 1
                break
            if x:
                added_just_one_constraint += 1
                if saving[0] == route[0] or saving[0] == route[len(route) - 1]:
                    first_external = 1
                    first_route_index = index
                if added_just_one_constraint == 2:
                    added_both_different_routes_constraint = 1
                    break
            if y:
                added_just_one_constraint += 1
                if saving[1] == route[0] or saving[1] == route[len(route) - 1]:
                    second_external = 1
                    second_route_index = index
                if added_just_one_constraint == 2:
                    added_both_different_routes_constraint = 1
                    break
        if saving[0] == 2 or saving[1] == 2:
            dff = 1
        if not added_both_same_route_constraint:
            # new route
            if added_just_one_constraint == 0:
                if len(routes) < n_trucks:
                    # create new route
                    new_route = [saving[0], saving[1]]
                    routes.append(new_route)
                    stores_to_be_added.remove(saving[0])
                    stores_to_be_added.remove(saving[1])
            # insert
            elif added_just_one_constraint == 1:
                if first_external:
                    rx = routes[first_route_index]
                    if len(rx) < capacity:
                        if rx[0] == saving[0]:
                            rx.insert(0, saving[1])
                        elif rx[len(rx) - 1] == saving[0]:
                            rx.append(saving[1])
                        else:
                            raise ArithmeticError('added_just_one_constraint1')
                        stores_to_be_added.remove(saving[1])
                elif second_external == 1:
                    rx = routes[second_route_index]
                    if len(rx) < capacity:
                        if rx[0] == saving[1]:
                            rx.insert(0, saving[0])
                        elif rx[len(rx) - 1] == saving[1]:
                            rx.append(saving[0])
                        else:
                            raise ArithmeticError('added_just_one_constraint1')
                        stores_to_be_added.remove(saving[0])
            # merge
            elif added_both_different_routes_constraint and first_external == 1 and second_external == 1:
                r1 = routes[first_route_index]
                r2 = routes[second_route_index]
                if len(r1) + len(r2) < capacity:
                    routes.remove(r1)
                    routes.remove(r2)
                    new_route = []
                    if r1[0] == saving[0] and r2[0] == saving[1]:
                        r1.reverse()
                        new_route = r1 + r2
                    elif r1[len(r1) - 1] == saving[0] and r2[0] == saving[1]:
                        new_route = r1 + r2
                    elif r1[0] == saving[0] and r2[len(r2) - 1] == saving[1]:
                        new_route = r2 + r1
                    elif r1[len(r1) - 1] == saving[0] and r2[len(r2) - 1] == saving[1]:
                        r2.reverse()
                        new_route = r1 + r2
                    else:
                        raise ArithmeticError('added_just_one_constraint1')
                    routes.insert(0, new_route)

    if len(stores_to_be_added):
        last_route = finalize_remaining_shops_routes(stores_to_be_added)
        some_routes = list(filter(lambda route_x: len(route_x) + len(last_route) <= capacity, routes))
        if len(some_routes):
            new_route = some_routes[0] + last_route
            routes.remove(some_routes[0])
            routes.append(new_route)
        else:
            routes.append(last_route)

    return routes


# NEIGHBORS ALGORITHM
def truck_routes(depth: int):
    depot = int(location[0][0])
    routes = [[depot]]

    n_routes = 0
    n_nodes = 0
    n_nodes_in_route = 0
    available_stores_ind = built_stores.copy()
    available_stores_ind.remove(0)
    prev = 0

    while n_nodes < len(built_stores) - 1:
        available_distance = list(map(lambda d: distance[prev][d], available_stores_ind))

        # choose the closest available store as next
        next_list = list(filter(lambda available_store: distance[prev][available_store] == min(available_distance),
                                available_stores_ind))
        if len(next_list) > 0:
            next_s = next_list[0]
        else:
            raise ValueError from None
        # remove chosen store from the available ones
        available_stores_ind.remove(next_s)
        # add chosen store in route
        routes[n_routes].append(next_s + 1)
        n_nodes += 1
        n_nodes_in_route += 1
        # now next has been visited so make it the starting point
        prev = next_s

        if n_nodes_in_route >= depth:
            # open new route and back to depot as starting point
            n_nodes_in_route = 0
            n_routes += 1
            prev = 0
            if n_nodes < len(built_stores) - 1:
                # startup a new route setting the depot as first
                routes.append([depot])

    return routes


def cost_of_route(this_route):
    cost = 0
    if len(this_route) == 0:
        return cost
    for i in range(len(this_route)):
        if i == len(this_route) - 1:
            # at the end of the route, so go back to depot
            cost += distance[this_route[i] - 1][0]
        else:
            cost += distance[this_route[i] - 1][this_route[i + 1] - 1]

    cost = fc + vc * cost

    return cost


def cost_of_route_Clarke(this_route, this_distance):
    if len(this_route) == 0 or len(this_route) == 1:
        return 0
    cost = 0
    for i in range(len(this_route) - 1):
        cost += this_distance[this_route[i]][this_route[i + 1]]
    cost += this_distance[this_route[len(this_route) - 1]][0]
    cost = fc + vc * cost
    return cost


# GLOBAL PROCEDURE

# Open files dat and read file

solution_path = "solution.txt"

if len(argv) > 1:
    datContent = [i.strip().split() for i in open(argv[1]).readlines()]
else:
    sys.exit("No input dataset file given! Please give the .dat file as argument.")

param = []
for i in datContent[:5]:
    param.append(int(i[3]))

# Variable declaration

location = datContent[7:len(datContent) - 1]

for i in location:
    for j in range(0, len(i)):
        i[j] = int(i[j])

n = int(param[0])
range_param = int(param[1])
vc = int(param[2])
fc = int(param[3])
capacity = int(param[4])

distance = []
for i in location:
    temp = []
    for j in location:
        temp.append(sqrt(pow(j[1] - i[1], 2) + pow(j[2] - i[2], 2)))
    distance.append(temp)

truck = []
built_stores = []

N = set(range(len(location)))
T = set(range(len(truck)))
model = Model()

# Variables
# 1 if store in city j is visited by truck i
x = [[model.add_var(var_type=BINARY) for j in N] for i in T]
# 1 if store is active in city i
y = [model.add_var(var_type=BINARY) for i in N]

# CONSTRAINTS
# Shop at location 1 is mandatory
model += y[0] == 1

# Activable only if usable attribute = 1
# model += (y[i] <= location1[i][4] for i in N)
for i in N:
    model += y[i] <= location[i][4]

# at least one store in range
for i in N:
    model += xsum(y[j] for j in N if distance[i][j] < range_param) >= 1

# objective function: minimize the cost of stores
model.objective = minimize(xsum(location[i][3] * y[i] for i in N))

model.optimize()

# Checking if a solution was found
if model.num_solutions:
    out.write('Shops to open with total cost of %g found.\n'
              % model.objective_value)
for i in range(len(y)):
    if y[i].x:
        built_stores.append(i)

# compute results of Neighbor algorithm
best_cost_Neighbor = 0
best_result_Neighbor = [[]]
try:
    best_result_Neighbor = truck_routes(capacity)
    for route in best_result_Neighbor:
        mid_cost = cost_of_route(route)
        best_cost_Neighbor += mid_cost
except ValueError as exp:
    print("Error: No built stores")

# repeat truck_routes for capacity down to 1 and check with cost_of_route if solution better the previous
for i in range(capacity - 1, 0, -1):
    result = truck_routes(i)
    total_cost = 0
    for route in result:
        mid_cost = cost_of_route(route)
        total_cost += mid_cost

    if total_cost < best_cost_Neighbor:
        best_cost_Neighbor = total_cost
        best_result_Neighbor = result


# compute results of Clarke-Wright algorithm
use_neighbor = False
best_cost_Clarke = 0
best_result_Clarke = [[]]
savings = get_savings(built_stores, distance)
n_truck = ceil((len(built_stores) - 1) / capacity)
try:
    routes = get_routes(built_stores, savings, capacity, n_truck)
    [route.insert(0, 0) for route in routes]
except IndexError:
    use_neighbor = True

if not use_neighbor:
    for store in built_stores:
        if store == 0:
            continue
        c = 0
        for route in routes:
            c += route.count(store)
        if c != 1:
            use_neighbor = True
            break

if not use_neighbor:
    try:
        best_result_Clarke = routes
        for route in best_result_Clarke:
            mid_cost = cost_of_route_Clarke(route, distance)
            best_cost_Clarke += mid_cost

        best_result_Clarke = [list(map(lambda store: store + 1, route)) for route in best_result_Clarke]
    except ValueError as exp:
        print("Error: No built stores")

best_cost = best_cost_Neighbor if best_cost_Neighbor < best_cost_Clarke or use_neighbor else best_cost_Clarke
best_result = best_result_Neighbor if best_cost == best_cost_Neighbor else best_result_Clarke
print("Neighbor cost: " + str(best_cost_Neighbor) + "\nClarke cost: " + str(best_cost_Clarke))
####

solution_total_cost = best_cost + model.objective_value
solution_opening_cost = model.objective_value
solution_refurbishing_cost = best_cost
opened_stores = list(map(lambda store_x: store_x+1, built_stores))
refurbishing_tracks = best_result


try:
    with open(solution_path, 'w') as solution_file:
        solution_file.write(str(solution_total_cost) + "\n")
        solution_file.write(str(solution_opening_cost) + "\n")
        solution_file.write(str(solution_refurbishing_cost) + "\n")
        solution_file.write(', '.join(map(str, opened_stores)))
        solution_file.write("\n")
        for track in refurbishing_tracks:
            track.append(1)  # returning to base
            solution_file.write(', '.join(map(str, track)))
            solution_file.write("\n")

        solution_file.close()
        print("Solution .txt generated in this directory.")
except FileNotFoundError:
    print("The 'docs' directory does not exist")
