# Recursive function used by countWays
def count_ways_recursive_function(n, m):
    if n <= 1:
        return n
    res = 0
    i = 1
    while i <= m and i <= n:
        res = res + count_ways_recursive_function(n - i, m)
        i = i + 1
    return res

# Returns number of ways to reach s'th stair
def count_ways(total_number_of_steps, m):
    return count_ways_recursive_function(total_number_of_steps + 1, m)

total_number_of_steps = 5
stair_climbing_number = [1, 2]

m = len(stair_climbing_number)


print ("Total number of ways to climb the stairs: " + str(count_ways(total_number_of_steps, m)))