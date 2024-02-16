"""
COMS 472 : AI - Lab 1
Author: Kausshik Manojkumar
"""
import sys
print(sys.executable)
from collections import deque
from queue import PriorityQueue
import argparse

from utils import *

# ______________________________________________________________________________

#Define a class for the 8-Puzzle problem -> Problem Class
#0 - is the blank tile

class EightPuzzle:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=(1,2,3,4,5,6,7,8,9,0)): # Constructor
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
        return sum(manhattan_distance(s, g) for (s, g) in zip(node.state, self.goal))
    
    def h3(self, node):
        """ Return the heuristic value for a given state. Gaschnig's heuristic"""
        return sum(gaschnig_distance(s, g) for (s, g) in zip(node.state, self.goal))

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
class SimpleProblemSolvingAgentProgram:
    """
    [Figure 3.1]
    Abstract framework for a problem-solving agent.
    """

    def __init__(self, initial_state=None):
        """State is an abstract representation of the state
        of the world, and seq is the list of actions required
        to get to a particular state from the initial state(root)."""
        self.state = initial_state
        self.seq = []

    def __call__(self, percept):
        """[Figure 3.1] Formulate a goal and problem, then
        search for a sequence of actions to solve it."""
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.state)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)

    def update_state(self, state, percept):
        raise NotImplementedError

    def formulate_goal(self, state):
        raise NotImplementedError

    def formulate_problem(self, state, goal):
        raise NotImplementedError

    def search(self, problem):
        raise NotImplementedError

# ______________________________________________________________________________
# SEARCH ALGORITHMS
def bfs(problem):
    """Breadth First Search"""
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None

def ids(problem, limit=50):
    """Iterative Deepening Search"""
    for depth in range(limit):
        result = dls(problem, depth)
        if result:
            return result
    return None

def dls(problem, limit):
    """Depth Limited Search"""
    node = Node(problem.initial)
    return recursive_dls(node, problem, limit)

def recursive_dls(node, problem, limit):
    """Recursive Depth Limited Search"""
    if problem.goal_test(node.state):
        return node
    elif limit == 0:
        return 'cutoff'
    else:
        cutoff_occurred = False
        for child in node.expand(problem):
            result = recursive_dls(child, problem, limit - 1)
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result
        return 'cutoff' if cutoff_occurred else None
    
def astar(problem, heuristic):
    """A* Search"""
    if heuristic == "h1":
        heuristic = problem.h1
    elif heuristic == "h2":
        heuristic = problem.h2
    else:
        heuristic = problem.h3

    node = Node(problem.initial)
    frontier = PriorityQueue('min', f=lambda node: node.path_cost + heuristic(node))
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if child.path_cost < incumbent.path_cost:
                    del frontier[incumbent]
                    frontier.append(child)
    return None


# ______________________________________________________________________________
# Main Method
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Solve an 8-puzzle problem with a specified algorithm.")
    parser.add_argument('--fPath', type=str, required=True, help='File path of the puzzle state')
    parser.add_argument('--alg', type=str, required=True, choices=['BFS', 'IDS', 'h1', 'h2', 'h3'], help='Algorithm to use for solving the puzzle')

    args = parser.parse_args()
    
    initial_state = read_puzzle_state(args.fPath)
    problem = EightPuzzle(initial=initial_state)

    if not problem.check_solvability(initial_state):
        print("The inputted puzzle is not solvable.")
        sys.exit(1)

    # Solve the puzzle based on the selected algorithm
    solution = None
    if args.alg == "BFS":
        solution = bfs(problem)
    elif args.alg == "IDS":
        solution = ids(problem)
    elif args.alg == "h1":
        solution = astar(problem, heuristic="h1")
    elif args.alg == "h2":
        solution = astar(problem, heuristic="h2")
    elif args.alg == "h3":
        solution = astar(problem, heuristic="h3")

    # Output the solution
    if solution:
        print("Solution found:", solution)
    else:
        print("No solution found.")