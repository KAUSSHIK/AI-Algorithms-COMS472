"""
COMS 472 : AI - Lab 1
Author: Kausshik Manojkumar
"""
import sys
from collections import deque
from queue import PriorityQueue
import argparse
import time
import itertools
import heapq
import math
import os

from utils import *

# ______________________________________________________________________________

#Define a class for the 8-Puzzle problem -> Problem Class
#0 - is the blank tile

class EightPuzzle:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=(1,2,3,4,5,6,7,8,0)): # Constructor
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal."""
        self.initial = initial
        self.goal = goal

    def find_blank_tile(self, state):
        return state.index(0);

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions."""
        possible_actions = ['U', 'D', 'L', 'R']
        blank_tile_index = self.find_blank_tile(state)

        if blank_tile_index % 3 == 0:
            possible_actions.remove('L')
        if blank_tile_index < 3:
            possible_actions.remove('U')
        if blank_tile_index % 3 == 2:
            possible_actions.remove('R')
        if blank_tile_index > 5:
            possible_actions.remove('D')
        
        return possible_actions

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        blank_tile_index = self.find_blank_tile(state)
        new_state = list(state) # Duplicate the state
        # Dictionary to map the action to the index change - really useful
        d = {'U': -3, 'D': 3, 'L': -1, 'R': 1}
        # Get the index of the neighbor tile to the blank tile
        neighbor_index = blank_tile_index + d[action]
        # Swap the blank tile with the neighbor tile
        new_state[blank_tile_index], new_state[neighbor_index] = new_state[neighbor_index], new_state[blank_tile_index]

        return tuple(new_state)
    
        

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor."""

        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1
    
    def check_solvability(self, state):
        """Check if the given state is SOLVABLE (helps with removing unsolvable states from the search space).
        There is a simple way to check if a puzzle is solvable or not, by counting
        the total number of “inversions” pairs.
        If the number is even, then the puzzle is solvable;
        otherwise, it is not"""
        num_inversions = 0
        for i in range(8): # 0 to 7
            for j in range(i+1, 9): # i+1 to 8
                if state[i] > state[j] and state[i] != 0 and state[j] != 0:
                    num_inversions += 1
        return num_inversions % 2 == 0
    

    def h1(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = number of misplaced tiles """
        return sum(s != g for (s, g) in zip(node.state, self.goal))
    
    def h2(self, node):
        """ Return the heuristic value for a given state. Manhattan distance heuristic function is used """   
        """Heuristic function that calculates the total Manhattan distance of a puzzle state from the goal state."""
        state = node.state
        goal = self.goal
        total_distance = 0
        for i in range(1, 9):  # Assuming 8-puzzle
            dx, dy = abs(state.index(i) % 3 - goal.index(i) % 3), abs(state.index(i) // 3 - goal.index(i) // 3)
            total_distance += dx + dy
        return total_distance
    
    def h3(self, node):
        """ Return the heuristic value for a given state. Gaschnig's heuristic"""
        """Heuristic function that calculates the total Euclidean distance of a puzzle state from the goal state."""
        state = node.state
        goal = self.goal
        total_distance = 0
        for i in range(1, 9):  # Assuming 8-puzzle
            dx, dy = abs(state.index(i) % 3 - goal.index(i) % 3), abs(state.index(i) // 3 - goal.index(i) // 3)
            total_distance += math.sqrt(dx * dx + dy * dy)
        return total_distance

    #Unused - might remove
    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
# ______________________________________________________________________________

#Define a class for the Node of the search tree -> Node Class
class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

# ______________________________________________________________________________
# SEARCH ALGORITHMS
def bfs(problem):
    print("BFS")
    start_time = time.time()
    initial_state = Node(problem.initial)
    frontier = deque([initial_state])
    explored = set()
    nodes_generated = 0

    while frontier:
        if(time.time() - start_time > 900):
            return (None, nodes_generated, 'timedout')
        
        current_node = frontier.popleft()
        explored.add(current_node)
        nodes_generated += 1

        if problem.goal_test(current_node.state):
            return (current_node, nodes_generated, time.time() - start_time)
        
        explored.add(current_node.state)

        for child in current_node.expand(problem):
            if child.state not in explored and child not in frontier:  # Check if the state has been explored
                frontier.append(child)

    return (None, nodes_generated, time.time() - start_time)

def ids(problem, limit=50):
    """Iterative Deepening Search"""
    print("IDS")
    start_time = time.time()
    nodes_generated = 0
    for depth in itertools.count():
        if(time.time() - start_time > 900):
            return (None, nodes_generated, 'timedout')
        if depth > limit:
            return (None, nodes_generated, time.time() - start_time)
        if(time.time() - start_time > 900):
            return (None, nodes_generated, 'timedout')
        result, new_nodes = dls(problem, depth)
        if(time.time() - start_time > 900):
            return (None, nodes_generated, 'timedout')
        nodes_generated += new_nodes
        if result not in ['cutoff', None]:
            return (result, nodes_generated, time.time() - start_time)
    return (None, nodes_generated, time.time() - start_time)

def dls(problem, limit):
    """Depth Limited Search"""
    node = Node(problem.initial)
    nodes_generated = 0  # Initialize node count
    result = recursive_dls(node, problem, limit, nodes_generated)
    return result

def recursive_dls(node, problem, limit, nodes_generated):
    """Recursive Depth Limited Search"""
    if problem.goal_test(node.state):
        return node, nodes_generated
    elif limit == 0:
        return 'cutoff', nodes_generated
    else:
        cutoff_occurred = False
        for child in node.expand(problem):
            nodes_generated += 1  # Increment for each child generated
            result, nodes_generated = recursive_dls(child, problem, limit - 1, nodes_generated)
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result, nodes_generated
        return ('cutoff' if cutoff_occurred else None), nodes_generated
    
def astar(problem, heuristic):
    print("A* " + ("h1" if heuristic == problem.h1 else ("h2" if heuristic == problem.h2 else "h3")))
    """A* Search"""
    def f(node):
        """A* heuristic function - cost from start to node + estimated cost from node to goal"""
        return node.path_cost + heuristic(node)
    
    start_time = time.time()
    node = Node(problem.initial)
    nodes_generated = 1
    if problem.goal_test(node.state):
        return (node, nodes_generated, time.time() - start_time)
    frontier = []
    heapq.heappush(frontier, (f(node), node))
    explored = set()

    while frontier:
        if(time.time() - start_time > 900):
            return (None, nodes_generated, 'timedout')
        node = heapq.heappop(frontier)[1]
        if problem.goal_test(node.state):
            return (node, nodes_generated, time.time() - start_time)
        explored.add(node.state)
        for child in node.expand(problem):
            nodes_generated += 1
            if child.state not in explored and child not in [item[1].state for item in frontier]:
                heapq.heappush(frontier, (f(child), child))
            elif child.state in [item[1].state for item in frontier]:
                index = [item[1].state for item in frontier].index(child.state)
                if f(child) < frontier[index][0]:
                    frontier[index] = (f(child), child)
                    heapq.heapify(frontier)
    return (None, nodes_generated, time.time() - start_time)



# ______________________________________________________________________________
# Main Method
def concat_list(lst):
    return ''.join(str(i) for i in lst)

def concatenate_modify_list(lst):
    # Join the characters into a string
    s = ''.join(lst)
    
    # Create a translation table
    trans = str.maketrans('UDLR', 'DURL')
    
    # Use the translation table to replace the characters
    return s.translate(trans)

def automate_part2():
    algorithms = ['BFS','IDS','h1', 'h2', 'h3']
    filepaths = ['./Part2/S1.txt', './Part2/S2.txt', './Part2/S3.txt', './Part2/S4.txt', './Part2/S5.txt']
    with open("./Part2Output.txt", "w") as outF:
        for f in filepaths:
            initial_state = read_puzzle_state(f)
            problem = EightPuzzle(initial=initial_state)
            if not problem.check_solvability(initial_state):
                s = "UNSOLVABLE"
                outF.write(f +": "+ s + "\n\n")
                continue
            for a in algorithms:
                if a == "BFS":
                    solution_node, nodes_generated, total_time = bfs(problem)
                elif a == "IDS":
                    solution_node, nodes_generated, total_time = ids(problem)
                elif a == "h1":
                    solution_node, nodes_generated, total_time = astar(problem, problem.h1)
                elif a == "h2":
                    solution_node, nodes_generated, total_time = astar(problem, problem.h2)
                elif a == "h3":
                    solution_node, nodes_generated, total_time = astar(problem, problem.h3)

                if solution_node is not None:
                    solution = solution_node.solution()
                elif solution_node is None and total_time == 'timedout':
                    solution = None
                else:
                    solution = None
                
                if solution is not None:
                    minutes, remainder = divmod(total_time, 60)
                    seconds, milliseconds = divmod(remainder, 1)
                    milliseconds *= 1000  # convert from seconds to milliseconds

                    actions = solution_node.solution()
                    path_length = len(actions)

                    s = f"Total nodes generated: {nodes_generated}\nTotal time taken: {int(minutes)} minutes {int(seconds)} seconds {int(milliseconds)} milliseconds\nPath length: {path_length}\nPath: {concatenate_modify_list(solution)}"
                elif solution is None and total_time == 'timedout':
                    s = f"Total nodes generated: {nodes_generated}\nTotal time taken >15 min\nPath length: Timed out.\nPath: Timed out."
                else:
                    s = "Total nodes generated: No solution found\nTotal time taken: No solution found\nPath length: No solution found\nPath: No solution found"
                
                algorithm = a if a in ["BFS", "IDS"] else f"A* {a}"
                outF.write(f"{f}: ALGORITHM USED => {algorithm} {s}\n")
            outF.write("\n\n")

def automate_part3():
    algorithms =  ['BFS','IDS','h1', 'h2', 'h3']
    folder_dirs = ['./Part3/L8/', './Part3/L15/', './Part3/L24/']
    with open("./Part3Output.txt", "w") as outF:
        for d in folder_dirs:
            for filename in os.listdir(d):
                if filename.endswith(".txt"):
                    initial_state = read_puzzle_state(d + filename)
                    problem = EightPuzzle(initial=initial_state)
                    if not problem.check_solvability(initial_state):
                        s = "UNSOLVABLE"
                        outF.write(filename +": "+ s + "\n\n")
                        continue
                    for a in algorithms:
                        if a == "BFS":
                            solution_node, nodes_generated, total_time = bfs(problem)
                        elif a == "IDS":
                            solution_node, nodes_generated, total_time = ids(problem)
                        elif a == "h1":
                            solution_node, nodes_generated, total_time = astar(problem, problem.h1)
                        elif a == "h2":
                            solution_node, nodes_generated, total_time = astar(problem, problem.h2)
                        elif a == "h3":
                            solution_node, nodes_generated, total_time = astar(problem, problem.h3)

                        if solution_node is not None:
                            solution = solution_node.solution()
                        elif solution_node is None and total_time == 'timedout':
                            solution = None
                        else:
                            solution = None
                        
                        if solution is not None:
                            minutes, remainder = divmod(total_time, 60)
                            seconds, milliseconds = divmod(remainder, 1)
                            milliseconds *= 1000


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solve an 8-puzzle problem with a specified algorithm.")
    parser.add_argument('--runPart2', action='store_true')
    parser.add_argument('--runPart3', action='store_true')
    parser.add_argument('--fPath', type=str, help='File path of the puzzle state')
    parser.add_argument('--alg', type=str, choices=['BFS', 'IDS', 'h1', 'h2', 'h3'], help='Algorithm to use for solving the puzzle')

    args = parser.parse_args()
    
    #print("File Path: ", args.fPath)
    #print("Algorithm: ", args.alg)

    if args.runPart2:
        automate_part2()
        sys.exit(0)

    initial_state = read_puzzle_state(args.fPath)
    #print("Initial State: ", initial_state)
    problem = EightPuzzle(initial=initial_state)
    #print(problem.check_solvability(initial_state))

    if not problem.check_solvability(initial_state):
       print("Total nodes generated: UNSOLVABLE")
       print("Total time taken: UNSOLVABLE")
       print("Path length: UNSOLVABLE")
       print("Path: UNSOLVABLE")
       sys.exit(1)

    # start_time = time.time()
    if args.alg == "BFS":
        #print("BFS is called")
        solution_node, nodes_generated, total_time = bfs(problem)
    elif args.alg == "IDS":
        #print("IDS is called")
        solution_node, nodes_generated, total_time = ids(problem)
    elif args.alg == "h1":
        #print("A* with h1 is called")
        solution_node, nodes_generated, total_time = astar(problem, problem.h1)
    elif args.alg == "h2":
        #print("A* with h2 is called")
        solution_node, nodes_generated, total_time = astar(problem, problem.h2)
    elif args.alg == "h3":
        #print("A* with h3 is called")
        solution_node, nodes_generated, total_time = astar(problem, problem.h3)

    # end_time = time.time()
    # total_time = end_time - start_time
    
    # Solve the puzzle based on the selected algorithm
    if solution_node is not None:
        solution = solution_node.solution()
    elif solution_node is None and total_time == 'timedout':
        solution = None
    else:
        solution = None    

    # Create a process to run the search algorithm
    # if process.is_alive():
    #     print("Total nodes generated: <<??>>")
    #     print("Total time taken >15 min")
    #     print("Path length: Timed out.")
    #     print("Path: Timed out.")
    #     process.terminate()

    if solution is not None:
        minutes, remainder = divmod(total_time, 60)
        seconds, milliseconds = divmod(remainder, 1)
        milliseconds *= 1000  # convert from seconds to milliseconds

        actions = solution_node.solution()
        path_length = len(actions)

        print("Total nodes generated: ", nodes_generated)
        print(f"Total time taken: {int(minutes)} minutes {int(seconds)} seconds {int(milliseconds)} milliseconds")
        print("Path length:", path_length)
        print("Path:", concatenate_modify_list(solution))
    elif solution is None and total_time == 'timedout':
        print("Total nodes generated: ", nodes_generated)
        print("Total time taken >15 min")
        print("Path length: Timed out.")
        print("Path: Timed out.")
    else:
       print("Total nodes generated: No solution found")
       print("Total time taken: No solution found")
       print("Path length: No solution found")
       print("Path: No solution found")
