from math import sqrt
from sys import stdout as out
from sys import argv
from mip import Model, xsum, minimize, BINARY

def truck_routes():
    paths = []
    stores = built_stores.copy()
    stores.remove(0)
    '''
    savings = [[]]
    temp_stores_dump = []
    decrease_order_savings_ids = [()]

    for i in range(len(stores)):
        for k in range(i+1, len(stores)):
            savings[i].append(fc + vc*(distance[0][i] + distance[k][0] + distance[i][k]))
        print("Savings[" + i + "]: " + str(savings[i]) + "\n")

    # sort decreasing order
    for i in range(len(savings)):
        temp_stores_dump.append(savings[i])
    '''

    savings2 = []
    for i in range(len(stores)):
        for k in range(i+1, len(stores)):
            savings2.append({
                'i': i,
                'k': k,
                'save': fc + vc*(distance[0][stores[i]] + distance[stores[k]][0] + distance[stores[i]][stores[k]]),
            })

    def take_save(elem):
        return elem['save']

    for i in range(len(savings2)):
        savings2.sort(reverse=True, key=take_save)
    print("Sorted: " + str(savings2))

    #start merging paths
    for saving in savings2:
        already_in_path = False
        for path in paths:

            if saving['i'] in path:
                already_in_path = True
                if len(path) >= capacity + 2:
                    # new single path
                    paths.append([0, saving['k'], 0])
                    break
                index = path.index(saving['i'])
                path.insert(index+1, saving['k'])
                break
            elif saving['k'] in path:
                already_in_path = True
                if len(path) >= capacity + 2:
                    # new single path
                    paths.append([0, saving['i'], 0])
                    break
                index = path.index(saving['k'])
                path.insert(index-1, saving['i'])
                break
        if not already_in_path:
            # new path
            paths.append([0, saving['i'], saving['k'], 0])

    return paths



def cost_of_route(this_route):
    cost = 0
    if len(this_route) == 0:
        return cost
    for i in range(len(this_route)):
        cost += distance[this_route[i]][this_route[i + 1]]

    cost = fc + vc * cost

    return cost

datFile1 = "datasets/minimart-I-50.dat"
datFile2 = "datasets/minimart-I-100.dat"
solution_file_path1 = "minimart-I-50-solution.txt"
solution_file_path2 = "minimart-I-100-solution.txt"

if len(argv) > 1:
    datContent = [i.strip().split() for i in open(argv[1]).readlines()]
else:
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
best_result = truck_routes()
print(best_result)
for route in best_result:
    best_cost += cost_of_route(route)

print("BEST FOUND AT THE END IS: " + str(best_cost) + "\nWith route: " + str(best_result))
