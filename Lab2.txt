random_numbers = [0.6, 0.61, 0.61, 0.63, 0.64, 0.64, 0.640, 0.650, 0.655, 0.66, 0.680, 0.680, 0.70, 0.705, 0.710, 0.715, 0.720, 0.720, 0.720, 0.730 ]
house_size = []
seed = 2423494 # Set seed as your SID 

for x in random_numbers:
  house_size.append (seed*x)

cost = [3, 3.2, 3.4, 3.4, 3.5, 3.7, 3.4, 3.7, 3.8, 3.9, 4.0, 3.95, 4.0, 4.2, 4.4, 4.3, 4.2, 4.1, 4.4, 4.6 ]

print (house_size)
print (cost)

# Your code to create a scatter plot - size vs cost
house_size= [1454096.4, 1478331.34, 1478331.34, 1526801.22, 1551036.16, 1551036.16, 1551036.16, 1575271.1, 1587388.57, 1599506.04, 1647975.9200000002, 1647975.9200000002, 1696445.7999999998, 1708563.27, 1720680.74, 1732798.21, 1744915.68, 1744915.68, 1744915.68, 1769150.6199999999]
cost=[3, 3.2, 3.4, 3.4, 3.5, 3.7, 3.4, 3.7, 3.8, 3.9, 4.0, 3.95, 4.0, 4.2, 4.4, 4.3, 4.2, 4.1, 4.4, 4.6]
plt.scatter(house_size, cost, color='blue', marker='o')
plt.xlabel("House Size")
plt.ylabel("Cost")
plt.title("House Size vs Cost")
