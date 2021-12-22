from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY

# Open files dat and read file

datFile = "datasets/minimart-I-50.dat"

datContent = [i.strip().split() for i in open(datFile).readlines()]

param = []
for i in datContent[:5]:
    param.append(int(i[3]))

# Variable declaration

location_temp = datContent[7:len(datContent)-1]
location = [[]]
for i in location:
    for j in range(len(i)):
        location.append(int(location_temp[i][j]))

for i in location:
    for j in range(len(i)):
        print(i[j] + " ")
    print("\n")

n = param[0]
range_param = param[1]
vc = param[2]
fc = param[3]
capacity = param[4]

distance = [[1, 2, 3],
            [4, 5],
            [6]]
truck = []

built_stores = []

# print(n, range_param, vc, fc, capacity, location)


def first_part():
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
    model += (y[i] <= location[i][3] for i in N)

    #
    '''for j in T:
        model += xsum(x[i][j] for i in N) <= capacity
        '''

    '''for i in N:
        model += xsum(x[i][j] * y[i] for j in T) == 1
    '''
    # at least one store in range
    for i in N:
        model += xsum(y[j] for j in N if distance[i][j] < range_param) >= 1

    # objective function: minimize the cost of stores
    model.objective = minimize(xsum(location[i][2]*y[i] for i in N))

    model.optimize()

    # checking if a solution was found
    if model.num_solutions:
        out.write('shops to open with total cost of %g found:'
                  % model.objective_value)
    for i in range(len(y)):
        if y[i]:
            built_stores.append(i)

    '''    nc = 0
        while True:
            nc = [i for i in V if x[nc][i].x >= 0.99][0]
            out.write(' -> %s' % places[nc])
            if nc == 0:
                break
        out.write('\n')
        '''


def second_part():
