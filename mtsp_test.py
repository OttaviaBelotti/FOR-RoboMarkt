from math import sqrt
from sys import stdout as out
from itertools import product

from mip import Model, xsum, minimize, BINARY

datFile = "datasets/minimart-I-50.dat"
datFile1 = "datasets/minimart-I-100.dat"
datFile2 = "datasets/minimart-l-5.dat"

datContent = [i.strip().split() for i in open(datFile).readlines()]

param = []

for i in datContent[:5]:
    param.append(int(i[3]))

vc = int(param[2])
fc = int(param[3])

# Variable declaration

location = datContent[7:len(datContent) - 1]

for i in location:
    for j in range(0, len(i)):
        i[j] = int(i[j])


def tsp(selected_stores, store_distance):
    # number of nodes and list of vertices
    n, V = len(store_distance), set(range(len(store_distance)))

    model2 = Model()

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[model2.add_var(var_type=BINARY) for j in V] for i in V]

    # continuous variable to prevent subtours: each city will have a
    # different sequential id in the planned route except the first one
    y = [model2.add_var() for i in V]

    # objective function: minimize the distance
    model2.objective = minimize(xsum(store_distance[i][j] * x[i][j] for i in V for j in V))

    # constraint : leave each city only once
    for i in V:
        model2 += xsum(x[i][j] for j in V - {i}) == 1

    # constraint : enter each city only once
    for i in V:
        model2 += xsum(x[j][i] for j in V - {i}) == 1

    # subtour elimination
    for (i, j) in product(V - {0}, V - {0}):
        if i != j:
            model2 += y[i] - (n + 1) * x[i][j] >= y[j] - n

    # optimizing
    model2.optimize()

    # checking if a solution was found
    sub_route = []
    if model2.num_solutions:
        out.write('route with total distance %g found: %s'
                  % (model2.objective_value, selected_stores[0]))
        nc = 0
        sub_route.append(selected_stores[0])
        while True:
            nc = [i for i in V if x[nc][i].x >= 0.99][0]
            out.write(' -> %s' % selected_stores[nc])
            sub_route.append(selected_stores[nc])
            if nc == 0:
                break
    else:
        print("no solution:\t + selected_stores")
        exit(1)
    return model2.objective_value, sub_route


def mtsp(route, distance_matrix):
    cost = 0
    route_model = []
    for sub_route in route:
        sub_dist = sub_distance(sub_route, distance_matrix)
        c, sub = tsp(sub_route, sub_dist)
        cost += c
        route_model.append(sub)
    return cost, route_model


def sub_distance(selected_stores, distance_matrix):
    sub_distance_matrix = []
    for i in set(range(len(distance_matrix))):
        if selected_stores.count(i) > 0:
            line = []
            for j in set(range(len(distance_matrix[i]))):
                if selected_stores.count(j) > 0:
                    line.append(distance_matrix[i][j])
            sub_distance_matrix.append(line)
    return sub_distance_matrix


def calc_distances():
    distance = []
    for i in location:
        temp = []
        for j in location:
            temp.append(sqrt(pow(j[1] - i[1], 2) + pow(j[2] - i[2], 2)))
        distance.append(temp)
    return distance


def cost_of_route(this_route, distance):
    if len(this_route) == 0 or len(this_route) == 1:
        return 0
    cost = 0
    for i in range(len(this_route)-1):
        cost += distance[this_route[i]][this_route[i + 1]]
    cost += distance[this_route[len(this_route)-1]][0]
    cost = fc + vc * cost
    return cost


distance = calc_distances()
route_cluster = [[0, 5, 23, 42, 35, 15, 41, 13, 11, 49, 39], [0, 44, 21, 8, 1]]
route_gen_short = [[0, 5, 8, 15, 13], [0, 44, 21, 49, 35, 39, 11, 41, 23, 42, 1]]
route_gen_long = [[0, 1, 8, 21, 44], [0, 13, 39, 23, 42, 11, 35, 41, 15, 49, 5]]

route_cluster_cost = 0
for sub_route in route_cluster:
    route_cluster_cost += cost_of_route(sub_route, distance)
print("route_cluster_cost:\t" + str(route_cluster_cost))

route_gen_short_cost = 0
for sub_route in route_gen_short:
    route_gen_short_cost += cost_of_route(sub_route, distance)
print("route_gen_short_cost:\t" + str(route_gen_short_cost))

route_gen_long_cost = 0
for sub_route in route_gen_long:
    route_gen_long_cost += cost_of_route(sub_route, distance)
print("route_gen_long_cost:\t" + str(route_gen_long_cost))