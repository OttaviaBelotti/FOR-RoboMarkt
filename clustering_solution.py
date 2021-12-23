from math import sqrt
from itertools import product, chain
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY


def truck_routes(depth: int):
    depot = int(location[0][0])
    routes = [[depot]]

    n_routes = 0
    n_nodes = 0
    n_nodes_in_route = 0
    available_stores_ind = built_stores.copy()
    available_stores_ind.remove(0)
    prev = 0

    print("Available stores: " + str(available_stores_ind))
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


# GLOBAL PROCEDURE

# Open files dat and read file

datFile = "datasets/minimart-I-50.dat"
datFile1 = "datasets/minimart-I-100.dat"

datContent = [i.strip().split() for i in open(datFile1).readlines()]

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
best_cost = 0
best_result = [[]]
try:
    best_result = truck_routes(capacity)
    print("Route option with capacity " + str(capacity) + ": ")
    print(best_result)
    for route in best_result:
        mid_cost = cost_of_route(route)
        best_cost += mid_cost
        print("Cost for route: " + str(mid_cost))
except ValueError as exp:
    print("Error: No built stores")
print("Best cost for " + str(capacity) + ": " + str(best_cost))

# repeat truck_routes for capacity down to 1 and check with cost_of_route if solution better the previous
for i in range(capacity - 1, 0, -1):
    print("Index: " + str(i))
    result = truck_routes(i)
    print("Route option with capacity " + str(i) + ": ")
    print(result)
    total_cost = 0
    for route in result:
        mid_cost = cost_of_route(route)
        total_cost += mid_cost
        print("Cost for route: " + str(mid_cost))

    if total_cost < best_cost:
        best_cost = total_cost
        best_result = result

print("BEST FOUND AT THE END IS: " + str(best_cost) + "\nWith route: " + str(best_result))


