# Finish A* search function that can find path from starting point to the end
# The robot starts from start position (0,0) and finds a path to end position (4, 5)
# In the maze, 0 is open path while 1 means wall (a robot cannot pass through wall)
# heuristic is provided

# example result:
# [[0, -1, -1, -1, -1, -1],
#  [1, -1, -1, -1, -1, -1],
#  [2, -1, -1, -1, -1, -1],
#  [3, -1,  8, 10, 12, 14],
#  [4,  5,  6,  7, -1, 15]]


maze = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]

heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

start = [0, 0] # starting position
end = [len(maze)-1, len(maze[0])-1] # ending position
cost = 1 # cost per movement

move = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right


### finish the A* search funciton below
def search(maze, start, end, cost, heuristic):

    result = [[-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]]
    move_number = 0
    result[start[0]][start[1]] = move_number
    cost_move = cost
    if start == end:
        return result

    else:
        while start != end:
            # move up
            if start[0] - 1 >= 0 and maze[start[0] - 1][start[1]] == 0 and result[start[0] - 1][start[1]] == -1:
                move_up = cost_move + heuristic[start[0] - 1][start[1]]
            else:
                move_up = cost_move + 10000

            # move down
            if start[0] + 1 <= len(maze)-1 and maze[start[0] + 1][start[1]] == 0 and result[start[0] + 1][start[1]] == -1:
                move_down = cost_move + heuristic[start[0] + 1][start[1]]
            else:
                move_down = cost_move + 10000

            # move left
            if start[1] - 1 >= 0 and maze[start[0]][start[1] -1] == 0 and result[start[0]][start[1] - 1] == -1:
                move_left = cost_move + heuristic[start[0]][start[1] - 1]
            else:
                move_left = cost_move + 10000

            # move right
            if start[1] + 1 <= len(maze[0])-1 and maze[start[0]][start[1] + 1] == 0 and result[start[0]][start[1] + 1] == -1:
                move_right = cost_move + heuristic[start[0]][start[1] + 1]
            else:
                move_right = cost_move + 10000

            move_list = [move_up, move_down, move_left, move_right]

            index_of_move = move_list.index(min(move_list))
            move_number = move_number + 1
            if index_of_move == 0:
                start[0] = start[0] - 1
                start[1] = start[1]
                result[start[0]][start[1]] = move_number

            elif index_of_move == 1:
                start[0] = start[0] + 1
                start[1] = start[1]
                result[start[0]][start[1]] = move_number

            elif index_of_move == 2:
                start[0] = start[0]
                start[1] = start[1] - 1
                result[start[0]][start[1]] = move_number

            elif index_of_move == 3:
                start[0] = start[0]
                start[1] = start[1]+ 1
                result[start[0]][start[1]] = move_number

        return result


result = search(maze, start, end, cost, heuristic)
print ('the maze run')
print (result)
