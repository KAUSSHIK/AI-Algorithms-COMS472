# AI-Algorithms-COMS472

## Lab 1: The 8 - Puzzle Problem
This Lab deals with writing code to solve 8-puzzle problems. The objective of the puzzle is to rearrange a given initial configuration (starting state) of 8 numbers on a 3 x 3 board into a final configuration (goal state) with a *minimum* number of actions.

## What is needed?
The goal is to solve the 8-puzzle problem using the following algorithms:
- Breadth-First Search (BFS)
- IDS (Iterative deepening DFS) (you may want to check for cycles during DFS)
- A* with misplaced title heuristic (h1).
- A* search using the Manhattan distance heuristic (h2).
- A* with one more heuristic (Gaschnig's heurisitc) (h3).

## Part 1
### Input
Two command line arguments: 1. File Path 2. Algorithm to use (BFS/IDS/h1/h2/h3).
### Output
1. Total nodes generated (for A* this includes nodes in closed list and fringe).
2. Total time taken.
3. A valid sequence of actions that will take the given state to the goal state.
a. Please note that the meaning of action is important (action is associated with the
movement of a tile, not with the movement of the blank space). For example, in
Fig 1 above, the action sequence is DRUL.
4. Also note that not all puzzles are solvable (See Appendix A1). The code should check
whether the puzzle is solvable or not before attempting to solve it.

## Part 2
Need to run code on all files given in folder 'Part2' and compile all your results into a single file, indicating which file/state leads to which output for each algorithm.

THIS HAS BEEN SOLVED, AND THE OUTPUT IS IN THE FILE `Part2Output.txt`

## Part 3
In this part, you will compare the performance of the algorithms you have coded.
1. In file part3.zip you’ll find 60 8-puzzles. 20 from each of 8, 15, and 24 levels, where the level indicates the optimal path length of the state from the goal.
2. For states in each level solve the puzzle and calculate the average run time and average
nodes generated for all the five algorithms.
3. Tabulate your results in the form of a table as shown below. Also, discuss conclusions drawn from the performance of different algorithm

THE LOG FOR THIS HAS BEEN GENERATED, AND THE OUTPUT IS IN THE FILE `Part3Output.txt`
The computation time and the number of nodes generated for each algorithm for each level of the puzzle has been calculated and tabulated below:

![image](https://github.com/KAUSSHIK/AI-Algorithms-COMS472/assets/32772050/c1376291-23a4-4f13-947b-37989103d0b2)


## HOW TO RUN THE CODE
1. Make sure you have python installed on your system.
2. cd into Lab1 directory using the command `cd Lab1`
3. Run the following command in the terminal:
   `python3 lab1.py --fPath <file_path> --alg <algorithm>`
   where `<file_path>` is the path to the file containing the initial state of the 8-puzzle problem and `<algorithm>` is the algorithm to be used to solve the problem. The algorithm can be one of the following: BFS, IDS, h1, h2, h3.
4. To generate output for Part 2, run the following command:
   `python3 lab1.py --runPart2`
5. To generate output for Part 3, run the following command:
    `python3 lab1.py --runPart3`
6. To watch the statistics of the algorithms, run the following command:
    `python3 calculateStats.py`
