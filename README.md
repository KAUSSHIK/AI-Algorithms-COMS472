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

## Solutions to Lab 1
- The solutions to the lab can be found in the `Lab1` folder.

## Lab 2: Gomoku
This Lab deals with writing code to implement an AI agent capable of playing Gomoku (Five-in-a-Row), a strategy board game played on a 15x15 grid. The agent uses an alpha-beta pruning algorithm to determine the optimal move based on the current game state and the chosen evaluation function.

## What is needed?
The goal is to implement the following components for the Gomoku AI:
- Alpha-Beta Agent: Utilizes the alpha-beta pruning algorithm to efficiently determine optimal moves by evaluating the game state up to a certain depth.
- Evaluation Functions: Two evaluation functions are implemented to assess the game state's favorability towards the agent. These functions consider factors like potential unbroken lines of stones and blocking opponent moves.
- Move Generator: Generates a list of legal moves from the current state, ensuring that moves adhere to Gomoku's rules.
- Game State Management (GUI): Manages the board's state, including stone placement and checking for game-ending conditions.

## Solutions to Lab 2
- The solutions to the lab can be found in the `Lab2` folder.