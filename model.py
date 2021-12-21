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

location = datContent[7:len(datContent)-1]
for i in location:
    for j in range(len(i)):
        i[j] = int(i[j])

n = param[0]
range_param = param[1]
vc = param[2]
fc = param[3]
capacity = param[4]

# distance = [[]]
truck = []

# print(n, range_param, vc, fc, capacity, location)

N = set(range(len(location)))
T = set(range(len(truck)))
model = Model()

# Variables
# 1 if store in city j is visited by truck i
x = [[model.add_var(var_type=BINARY) for j in N] for i in T]
# 1 if store is active in city i
y = [model.add_var(var_type=BINARY) for i in N]

#Constraints
model += y[1] == 1
model += (y[i] <= location[i][3] for i in N)

for j in T:
    model += xsum(x[i][j] for i in N) <= capacity

for i in N:
    model += xsum(x[i][j] * y[i] for j in T) == 1

#model += xsum(y[j] for (j in N, distance[i,j]<range_param)) >= 1

