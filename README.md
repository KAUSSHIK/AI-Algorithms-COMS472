# AI-Algorithms-COMS472

## Lab 1: The 8 - Puzzle Problem
This Lab deals with writing code to solve 8-puzzle problems. The objective of the puzzle is to rearrange a given initial configuration (starting state) of 8 numbers on a 3 x 3 board into a final configuration (goal state) with a *minimum* number of actions.

## What is needed?
The goal is to solve the 8-puzzle problem using the following algorithms:
- Breadth-First Search (BFS)
- IDS (Iterative deepening DFS) (you may want to check for cycles during DFS)
- A* with misplaced title heuristic (h1).
- A* search using the Manhattan distance heuristic (h2).
- A* with one more heuristic (Euclidean/some other heuristic) (h3).

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

## Part 3
In this part, you will compare the performance of the algorithms you have coded.
1. In file part3.zip you’ll find 60 8-puzzles. 20 from each of 8, 15, and 24 levels, where the level indicates the optimal path length of the state from the goal.
2. For states in each level solve the puzzle and calculate the average run time and average
nodes generated for all the five algorithms.
3. Tabulate your results in the form of a table as shown below. Also, discuss conclusions drawn from the performance of different algorithm
