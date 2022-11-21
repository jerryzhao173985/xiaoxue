# An example of a problem formulation that uses the xpress.Dot() operator
# to formulate constraints simply. Note that the NumPy dot operator is not
# suitable here as the result is an expression in the Xpress variables.

import xpress as xp
import numpy as np

import pandas as pd
import numpy as np
import os

def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)

charg_point_path = os.path.join('Project_data/Charging_points.xlsx')
charg_point_df = pd.read_excel(charg_point_path)

demand_path = os.path.join('Project_data/Demand_data.xlsx')
demand_df = pd.read_excel(demand_path)

# points of interest in each grid
interest_point_path = os.path.join('Project_data/Interest _points.xlsx')
interest_point_df = pd.read_excel(interest_point_path)

amenity1 = set(interest_point_df['amenity'])

# potential charging points
pt_charg_point_path = os.path.join('Project_data/Potential_charging_points.xlsx')
pt_charg_point_df = pd.read_excel(pt_charg_point_path)

amenity2 = set(interest_point_df['amenity'])

## DEFINE
number_of_grids = 434
number_of_years = 4
grids = range(number_of_grids)
years = range(number_of_years)

# the demand every slow/ fast/ rapid chargers can satisfy
slow_c = [2000, 3500]
slow_avg = (slow_c[0]+slow_c[1])/2
fast_c = [4000, 5200]
rapid_c = [30000, 50500]
fast_avg = (fast_c[0]+fast_c[1])/2
rapid_avg = (rapid_c[0]+rapid_c[1])/2

## for visualizations
# print(charg_point_df.columns)
charg_point_index = charg_point_df['grid number'].to_numpy()
## for visualizations
# print(interest_point_df.columns)
interest_point_index = interest_point_df['grid number'].to_numpy()
## for visualizations
# print(pt_charg_point_df.columns)
pt_charg_point_index = pt_charg_point_df['grid number'].to_numpy()

neighbor = demand_df['NEIGHBORS'].to_numpy()
neighbor = [np.array(ne[1:-1].replace(' ', '').split(','), dtype = int) for ne in neighbor]

num_potential = demand_df['Number of Potential Locations'].to_numpy()
num_interest = demand_df['Number of PoI'].to_numpy()

exist_cp = demand_df['Number of Charging Points'].to_numpy()
exist_slow =  demand_df['Number of Slow Charging Points'].to_numpy()
exist_fast = demand_df['Number of Fast Charging Points'].to_numpy()
exist_rapid = demand_df['Number of Rapid Charging Points'].to_numpy()
demand_0 = demand_df['Demand_0'].to_numpy()

## potential_list >= intereset_list
## existing_list ?
# pt_charg_point_index, interest_point_index, charg_point_index

## Lets see if the point of interest is in the potential charging points list
## if so, then we can easily put charging point there aka. place of interest
## if not, we find a point in the neighboring area that is in the potential charging point list
## also we should consider energy efficiency at this point/ points
construct = {}
for k in interest_point_index:
    if k in pt_charg_point_index:
        construct[k] = [k]
#     if k not in pt_charg_point_index:
    else:
        construct[k] = []
        neighboring_area = neighbor[k-1]
        for nei in neighboring_area:
            if nei in pt_charg_point_index:
                construct[k].append(nei)

##enlarge construct so that include itself along with its neighbors
# for i in range(number_of_grids):
#     ii = i+1
#     if ii not in construct.keys():
#         construct[ii] = [ii]
#     else:
#         construct[ii].append(ii)

# print(construct)
# for i in range(number_of_grids):
#     ii = i+1
#     if ii not in construct.keys():
#         construct[ii] = []

# here A is a matrix 434*434 denoting at each index if the other relevant index is neighbor(1) or not(0)
# A is a binary matrix
A = np.zeros((number_of_grids, number_of_grids))  # A is a 434*434 matrix
for i in range(number_of_grids):
    for j in range(number_of_grids):
        if (i+1) in construct.keys():
            if (j+1) in construct[i+1]:
                A[i,j] = 1

print(find_indices(A[192, :],1))
print(construct[193])    

# I = np.eye(number_of_grids)  # I is the identity matrix

# Create a NumPy array of variables by using the xp.vars() function
x = xp.vars(number_of_grids, vartype=xp.integer)
# x0 = np.random.random(number_of_grids)  # random vector

# 6 constraints (rows of A)
tolerance = 0
Lin_sys1 = xp.Dot(A, (x+exist_cp)) >= np.array(num_interest)-tolerance
Lin_sys2 = x<= np.array(num_potential)
Lin_sys3 = xp.Dot(A, (x+exist_cp)) >= np.array(demand_0)/slow_avg

print(xp.Dot(A, x)[261])
print(construct[262])
print(xp.Dot(A, x)[397])
print(construct[398])
# print(Lin_sys1)


# One quadratic constraint
# A = np.random.random(30).reshape(6, 5)  # A is a 6x5 matrix
# Q = np.random.random(25).reshape(5, 5)  # Q is a 5x5 matrix
# Conv_c = xp.Dot(x, Q, x) <= 1

p = xp.problem()

p.addVariable(x)
p.addConstraint(Lin_sys1,Lin_sys2)
# p.delConstraint(213) # deletes R214/R648
# p.delConstraint(149)
p.delConstraint(789)
p.delConstraint(723)
p.delConstraint(647)
p.delConstraint(597)


# p.setObjective(xp.Dot(x-x0, x-x0))
# obj = xp.Dot(A, x) - np.array(num_interest)
# print("length of the not satisfied units: ", len(obj))
p.setObjective(xp.Sum(x), sense=xp.minimize)
# m.setObjective(xp.Sum ([y[i]⁎⁎2 for i in range (10)]))
# objective overwritten at each setObjective()  m.setObjective(xp.Sum([i*v[i] for i in S]), sense=xp.minimize)
p.solve()

# In order to investigate why the solver considers your model infeasible, 
# you can use function problem.iisfirst which computes a minimal set of constraints that make your problem infeasible. 
# Once the function has finished, you can use problem.iisgetdata to get the list of those constraints.
 
p.iisfirst(0)  # This looks for the first IIS. 
# p.iisgetdata()

print(p.getSolution ())            # prints a list with an optimal solution