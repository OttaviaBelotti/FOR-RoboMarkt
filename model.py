from math import sqrt
from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY

# Open files dat and read file

datFile = "datasets/minimart-I-50.dat"
datFile1 = "datasets/minimart-I-100.dat"


datContent = [i.strip().split() for i in open(datFile).readlines()]

param = []
for i in datContent[:5]:
    param.append(int(i[3]))

# Variable declaration

location = datContent[7:len(datContent)-1]

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
        temp.append(sqrt(pow(j[1]-i[1], 2) + pow(j[2]-i[2], 2)))
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
model.objective = minimize(xsum(location[i][3]*y[i] for i in N))

model.optimize()

# Checking if a solution was found
if model.num_solutions:
    out.write('shops to open with total cost of %g found:'
              % model.objective_value)
for i in range(len(y)):
    if y[i].x:
        built_stores.append(i)

print(built_stores)