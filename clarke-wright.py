import bisect
from math import sqrt, ceil
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY


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


def truck_routes(stores, depth: int):
    depot = int(location[0][0])
    routes = []

    n_routes = 0
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
        luca = truck_routes(stores_to_be_added, 1)
        some_routes = list(filter(lambda route_x: len(route_x) + len(luca) <= capacity, routes))
        if len(some_routes):
            new_route = some_routes[0] + luca
            routes.remove(some_routes[0])
            routes.append(new_route)
        else:
            routes.append(luca)

    return routes


def cost_of_route(this_route, distance):
    if len(this_route) == 0 or len(this_route) == 1:
        return 0
    cost = 0
    for i in range(len(this_route) - 1):
        cost += distance[this_route[i]][this_route[i + 1]]
    cost += distance[this_route[len(this_route) - 1]][0]
    cost = fc + vc * cost
    return cost


# GLOBAL PROCEDURE

# Open files dat and read file

datFile0 = "datasets/minimart-l-5.dat"
datFile1 = "datasets/minimart-I-50.dat"
datFile2 = "datasets/minimart-I-100.dat"
solution_file_path1 = "minimart-I-50-solution_clarke.txt"
solution_file_path2 = "minimart-I-100-solution_clarke.txt"

datContent = [i.strip().split() for i in open(datFile2).readlines()]
solution_file_path = solution_file_path2

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
    out.write('shops to open with total cost of %g found:'
              % model.objective_value)
for i in range(len(y)):
    if y[i].x:
        built_stores.append(i)

print(built_stores)
print(len(built_stores))
best_cost = 0
best_result = [[]]
savings = get_savings(built_stores, distance)
n_truck = ceil((len(built_stores) - 1) / capacity)
routes = get_routes(built_stores, savings, capacity, n_truck)
# routes = get_routes2(savings, capacity, n_truck)
[route.insert(0, 0) for route in routes]
print(savings)
print(routes)
for store in built_stores:
    c = 0
    for route in routes:
        c += route.count(store)
    if c == 0:
        print(store)
try:
    best_result = routes
    print(best_result)
    for route in best_result:
        mid_cost = cost_of_route(route, distance)
        best_cost += mid_cost
        print("Cost for route: " + str(mid_cost))
except ValueError as exp:
    print("Error: No built stores")
print("Best cost for " + str(routes) + ": " + str(best_cost))

solution_total_cost = best_cost + model.objective_value
solution_opening_cost = model.objective_value
solution_refurbishing_cost = best_cost
opened_stores = list(map(lambda store: store + 1, built_stores))
refurbishing_tracks = [list(map(lambda store: store + 1, route)) for route in best_result]

try:
    with open(solution_file_path, 'w') as solution_file:
        solution_file.write(str(solution_total_cost) + "\n")
        solution_file.write(str(solution_opening_cost) + "\n")
        solution_file.write(str(solution_refurbishing_cost) + "\n")
        solution_file.write(', '.join(map(str, opened_stores)))
        solution_file.write("\n")
        for track in refurbishing_tracks:
            track.append(1)  # returning to base
            solution_file.write(', '.join(map(str, track)))
            # solution_file.write(str(track) + "\n")
            solution_file.write("\n")

        solution_file.close()
except FileNotFoundError:
    print("The 'docs' directory does not exist")
