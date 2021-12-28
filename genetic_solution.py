# genetic algorithm search of the one max optimization problem
import numpy as np
from sys import stdout as out
from numpy import sqrt
from numpy.random import randint
from numpy.random import rand
from mip import Model, xsum, minimize, BINARY
from itertools import product

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


# objective function
def objective_function(length):
    return 1 / (1 + length)


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
    c1 = np.append(p1[:i] + p2[i:])
    c2 = np.append(p2[:i] + p1[i:])
    return [c1, c2]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        for i in pt:
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
def genetic_algorithm(objective, stores, distances, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = 0, objective(pop[0])
    # enumerate generations
    for gen in range(n_iter):
        # evaluate all candidates in the population
        scores = [objective(c) for c in pop]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
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
    return [best, best_eval]


# define the total iterations
n_iter = 100
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
best, score = genetic_algorithm(objective_function, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
