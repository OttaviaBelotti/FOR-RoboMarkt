# genetic algorithm search of the one max optimization problem
import random

import numpy as np
from sys import stdout as out
from numpy import sqrt
from numpy.random import randint
from numpy.random import rand
from mip import Model, xsum, minimize, BINARY
from itertools import product
from math import ceil

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

    # Python3 program for the above approach
    from itertools import combinations
    import bisect


    # Function to count all the possible
    # combinations of K numbers having
    # sum equals to N
    def count_ways_util(n, k, sum, dp, arr, sub):
        # Base Cases
        if sum == n and k == 0:
            if len(arr) == 0:
                arr.append(sub)
                return 1
            i = bisect.bisect(arr, sub)
            if arr[i - 1] != sub:
                arr.insert(i, sub)
            return 1

        if sum >= n and k >= 0:
            return 0

        if k < 0:
            return 0

        # If the result is already memoised
        if dp[sum][k] != -1:
            if dp[sum][k] == 1:
                bisect.insort(sub, n - sum)
                i = bisect.bisect(arr, sub)
                if arr[i - 1] != sub:
                    arr.insert(i, sub)
            return dp[sum][k]

        # Recursive Calls
        cnt = 0
        for i in range(1, n + 1):
            sub_copy = sub.copy()
            bisect.insort(sub_copy, i)
            cnt += count_ways_util(n, k - 1, sum + i, dp, arr, sub_copy)

        # Returning answer
        dp[sum][k] = cnt
        return dp[sum][k]


    def route_size_combinations(n, k):
        dp = [[-1 for _ in range(k + 1)]
              for _ in range(n + 1)]
        arr = []
        sub = []
        c = count_ways_util(n, k, 0, dp, arr, sub)
        return c, arr


def calc_stores(location, distance, range_param):
    N = set(range(len(location)))
    model = Model()
    # Variables
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
    built_stores = []
    for i in range(len(y)):
        if y[i].x:
            built_stores.append(i)
    return building_cost, built_stores


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


def cost_of_route(this_route, distance):
    if len(this_route) == 0 or len(this_route) == 1:
        return 0
    cost = 0
    for i in range(len(this_route) - 1):
        cost += distance[this_route[i]][this_route[i + 1]]
    cost += distance[this_route[len(this_route) - 1]][0]
    cost = fc + vc * cost
    return cost


def cost_of_multiple_routes(this_route, distance):
    cost = 0
    for i in this_route:
        cost += cost_of_route(i, distance)
    return cost, this_route


# objective function
def objective_function(length):
    return length


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def single_point_crossover(p1, p2, i):
    # perform crossover
    c1 = np.append(p1[:i], p2[i:])
    c2 = np.append(p2[:i], p1[i:])
    return [c1, c2]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        for i in range(pt):
            c1, c2 = single_point_crossover(c1, c2, i)

    return [c1, c2]


# mutation operator
def mutation(p, r_mut):
    if rand() < r_mut:
        i = randint(0, len(p))
        j = randint(0, len(p))
        temp = p[i]
        p[i] = p[j]
        p[j] = temp
    return p


# todo
# genetic algorithm
def genetic_algorithm(objective, mtsp, stores, distances, capacity, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    stores_no_depot = stores[1:len(stores)]
    n_truck = ceil(len(stores_no_depot) / capacity)
    c, size_combinations = route_size_combinations(len(stores_no_depot), n_truck)
    size_combinations = list(filter(lambda comb: all(x <= capacity for x in comb), size_combinations))

    start = 0
    start_route = []
    for j in size_combinations[0]:
        sub_route = stores_no_depot[start:start + j]
        sub_route.insert(0, stores[0])
        start_route.append(sub_route)
        start += j

    best_cost, best = mtsp(start_route, distances)
    best_eval = objective(best_cost)

    combination_counter = 0
    for combination in size_combinations:
        combination_counter += 1
        print("combination_counter:\t" + str(combination_counter))
        route = []
        pop = []
        for i in range(n_pop):
            pop.append(random.sample(stores_no_depot, len(stores_no_depot)))
            pop_route = []
            start = 0
            for j in combination:
                sub_route = pop[i][start:start + j]
                sub_route.insert(0, stores[0])
                pop_route.append(sub_route)
                start += j
            route.append(pop_route)
        # enumerate generations
        for gen in range(n_iter):
            print("gen_counter:\t" + str(gen))
            # evaluate all candidates in the population
            costs = []
            evaluations = []
            for c in route:
                cost, route_model = mtsp(c, distances)
                costs.append(cost)
                evaluation = objective(cost)
                evaluations.append(evaluation)
                if evaluation < best_eval:
                    print("previous best:\t" + str(best))
                    print("previous best eval:\t" + str(best_eval))
                    print("total cost:\t" + str(best_cost))
                    best, best_eval, best_cost = route_model, evaluation, cost
                    print("new best:\t" + str(best))
                    print("new best eval:\t" + str(best_eval))
                    print("total cost:\t" + str(best_cost))

            # select parents
            selected = [selection(pop, evaluations) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in crossover(p1, p2, r_cross):
                    # mutation
                    mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
    return [best, best_cost]


# define the total iterations
n_iter = 2000
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 0.9
# perform the genetic algorithm search

building_cost, built_stores = calc_stores(location, distance, range_param)
print(str(built_stores))
print(str(built_stores))

best, score = genetic_algorithm(objective_function, cost_of_multiple_routes, built_stores, distance, capacity, n_iter,
                                n_pop, r_cross,
                                r_mut)

print('Done!')
print(str(best) + ":\t" + str(score))
