# Gomoku AI

This project implements an AI agent capable of playing Gomoku (Five-in-a-Row), a strategy board game played on a 15x15 grid. The agent uses an alpha-beta pruning algorithm to determine the optimal move based on the current game state and the chosen evaluation function.

## Game Rules

1. Players alternate turns placing a stone of their color (white or black) on an empty intersection on the board.
2. Black plays first, with the first stone placed in the center of the board.
3. The second player's first stone may be placed anywhere.
4. The first player's second stone must be placed at least three intersections away from the first stone.
5. The winner is the first to form an unbroken line of five stones horizontally, vertically, or diagonally.
6. The game ends in a draw if the board is filled without any player forming a line of 5 stones.

## Components

- Alpha-Beta Agent: Utilizes the alpha-beta pruning algorithm to efficiently determine optimal moves by evaluating the game state up to a certain depth.
- Evaluation Functions: Two evaluation functions are implemented to assess the game state's favorability towards the agent. These functions consider factors like potential unbroken lines of stones and blocking opponent moves.
- Move Generator: Generates a list of legal moves from the current state, ensuring that moves adhere to Gomoku's rules.
- Game State Management (GUI): Manages the board's state, including stone placement and checking for game-ending conditions.

## Evaluation Functions

1. `compute_utility`: This function evaluates the board state by considering various factors such as immediate wins, open and semi-open rows, potential winning moves, board control, and a dynamic scoring system based on game progress. It assigns higher scores to favorable patterns and adjusts the importance of certain factors based on the stage of the game.

2. `compute_utility_2`: This function focuses on defensive play by prioritizing blocking the opponent's potential winning moves and evaluating the overall defensive structure of the board. It assigns higher weights to defensive patterns and aims to prevent the opponent from forming winning lines.

The two evaluation functions differ in their approach and prioritization. While `compute_utility` takes a balanced approach considering both offensive and defensive aspects, `compute_utility_2` emphasizes defensive play and prioritizes blocking the opponent's potential wins.

## Usage

1. Make sure you have Python installed on your system.
2. Clone the repository and navigate to the project directory.
3. Run the `gomoku.py` script to start the game.
4. Follow the on-screen instructions to play against the AI agent.
5. To adjust the search depth of the AI agent, modify the `depth` variable in the `main()` function.

## Dependencies

- Python 3.x
- NumPy
