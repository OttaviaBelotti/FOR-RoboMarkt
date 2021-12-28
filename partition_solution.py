import collections
from math import sqrt, ceil
from itertools import product
import sys
import pprint
from sys import stdout as out
import multiprocessing as mp

import partition as partition
from mip import Model, xsum, minimize, BINARY
from more_itertools.more import set_partitions, partitions

datFile = "datasets/minimart-I-50.dat"
datFile1 = "datasets/minimart-I-100.dat"
datFile2 = "datasets/minimart-l-5.dat"

datContent = [i.strip().split() for i in open(datFile).readlines()]

param = []
for i in datContent[:5]:
    param.append(int(i[3]))

# Variable declaration

location = datContent[7:len(datContent) - 1]

for i in location:
    for j in range(0, len(i)):
        i[j] = int(i[j])

n = param[0]
range_param = param[1]
vc = param[2]
fc = param[3]
capacity = param[4]

distance = []
for i in location:
    temp = []
    for j in location:
        temp.append(sqrt(pow(j[1] - i[1], 2) + pow(j[2] - i[2], 2)))
    distance.append(temp)

truck = []
built_stores = []
routes = []

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
building_cost = model.objective_value
for i in range(len(y)):
    if y[i].x:
        built_stores.append(i)

print(built_stores)


def algorithm_u(list_of_elements, number_of_partitions, max_partition_size):
    def visit(n, a):
        ps = [[] for i in range(number_of_partitions)]
        for j in range(n):
            ps[a[j + 1]].append(list_of_elements[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            valid = 1
            for value in collections.Counter(a).values():
                if value > max_partition_size:
                    valid = 0
                    break
            if valid:
                yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            valid = 1
            for value in collections.Counter(a).values():
                if value > max_partition_size:
                    valid = 0
                    break
            if valid:
                yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                valid = 1
                for value in collections.Counter(a).values():
                    if value > max_partition_size:
                        valid = 0
                        break
                if valid:
                    yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                valid = 1
                for value in collections.Counter(a).values():
                    if value > max_partition_size:
                        valid = 0
                        break
                if valid:
                    yield visit(n, a)
                a[nu] = a[nu] + 1
            valid = 1
            for value in collections.Counter(a).values():
                if value > max_partition_size:
                    valid = 0
                    break
            if valid:
                yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            valid = 1
            for value in collections.Counter(a).values():
                if value > max_partition_size:
                    valid = 0
                    break
            if valid:
                yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    list_size = len(list_of_elements)
    a = [0] * (list_size + 1)  # a is a list of 0 with length list_size +1model.py:155
    for j in range(1, number_of_partitions + 1):
        a[list_size - number_of_partitions + j] = j - 1
    return f(number_of_partitions, list_size, 0, list_size, a)


def pretty_print(parts):
    print(']\n['.join('] ['.join(','.join(str(e) for e in loe) for loe in part) for part in parts))


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
    if model2.num_solutions:
        out.write('route with total distance %g found: %s'
                  % (model2.objective_value, selected_stores[0]))
        nc = 0
        sub_route = [selected_stores[0]]
        while True:
            nc = [i for i in V if x[nc][i].x >= 0.99][0]
            out.write(' -> %s' % selected_stores[nc])
            sub_route.append(selected_stores[nc])
            if nc == 0:
                break
    else:
        print("no solution:\t + selected_stores")
        exit(1)
    routes.append(sub_route)
    return model2.objective_value


def sub_distance(selected_stores, distance_matrix):
    sub_distance_matrix = []
    for i in N:
        if selected_stores.count(i) > 0:
            line = []
            for j in N:
                if selected_stores.count(j) > 0:
                    line.append(distance_matrix[i][j])
            sub_distance_matrix.append(line)
    return sub_distance_matrix


visitable_stores = built_stores.copy()
visitable_stores.remove(0)
partitions = algorithm_u(visitable_stores, ceil(len(visitable_stores) / capacity), capacity)
pretty_print(partitions)

refurbishment_cost = sys.float_info.max
best_partition = []
for part in partitions:
    cost = 0
    routes = []
    for route in part:
        route.insert(0, built_stores[0])
        dist = sub_distance(route, distance)
        cost += tsp(route, dist)
    if cost < refurbishment_cost:
        refurbishment_cost = cost
        best_partition = routes
        print("new best routes founf:\t"+str(best_partition))

maintenance_cost = fc * ceil(len(visitable_stores) / capacity) + vc * refurbishment_cost

print(distance)
print(built_stores)
print(maintenance_cost)
print(best_partition)
