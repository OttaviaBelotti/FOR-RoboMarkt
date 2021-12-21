from itertools import product
from sys import stdout as out
from mip import Model, xsum, minimize, BINARY

location = [[]]
distance = [[]]
truck = []
n = 50
range_param = 7
vc = 1
fc = 50
capacity = 10

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



