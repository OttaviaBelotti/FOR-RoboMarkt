from math import sqrt
from sys import stdout as out
from sys import argv
from mip import Model, xsum, minimize, BINARY


datFile1 = "datasets/minimart-I-50.dat"
datFile2 = "datasets/minimart-I-100.dat"
solution_file_path1 = "minimart-I-50-solution.txt"
solution_file_path2 = "minimart-I-100-solution.txt"

if len(argv) > 1:
    datContent = [i.strip().split() for i in open(argv[1]).readlines()]
else:
    datContent = [i.strip().split() for i in open(datFile2).readlines()]

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
# T = set(range(len(truck)))

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
for i in range(len(y)):
    if y[i].x:
        built_stores.append(i)

'''second part'''
C = set(range(len(built_stores)))
K = set(range(len(built_stores)))  # at max we have a truck for each store to refurbish
model_VR = Model()

# 1 if store in city i is included in the route of truck j
x = [[model_VR.add_var(var_type=BINARY) for j in K] for i in C]

# 1 if truck h goes from store i to store j
k = [[[model_VR.add_var(var_type=BINARY) for i in C] for j in C] for h in K]

# c1) each built shop must be included in exactly 1 route
for i in C:
    model_VR += (xsum(xsum(k[i][j][h] for j in C) for h in K)) == 1

# c2) if i is in route of h, then h must be going from j to another node j
for i in C:
    for h in K:
        model_VR += (xsum(k[i][j][h] for j in C-{i})) == x[i][h]
        model_VR += (xsum(k[j][i][h] for j in C-{i})) == x[i][h]

# c3) in a route, there can be at max capacity shops
'''
#questi sono constraints che ho provato ad aggiungere ma rendono il problema infeasible
for h in K:
    # tra l'altro questo non va proprio
    model_VR += xsum((k[i][j][h] for j in C-{i}) for i in C) <= (capacity+2)


# c4) there must be starting from 0 and finishing in 0
for h in K:
    model_VR += xsum(k[0][j][h] for j in C) == 1
    model_VR += xsum(k[i][0][h] for i in C) == 1

# no self loops
for h in K:
    for i in C:
        model_VR += k[i][i][h] == 0
'''


model_VR.objective = minimize(xsum(vc * xsum(xsum(distance[i][j]*k[i][j][h] for j in C) for i in C) + fc for h in K))

model_VR.optimize()

if model_VR.num_solutions:
    out.write('refurbishing total cost of: ')
    out.write(str(model_VR.objective_values))