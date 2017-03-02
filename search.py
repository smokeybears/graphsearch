# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

#  python pacman.py -l mediumDottedMaze -p SearchAgent -a fn=ucs 
# python pacman.py -l tinyMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def graph_search(queue, problem):
    explored = set()
    fringe = queue
    
    start_node = problem.getStartState()
    enqueue(fringe, start_node, problem.getSuccessors(start_node))
    while not fringe.isEmpty():
        current_node = fringe.pop()
        if current_node[0] not in explored: 
            explored.add(current_node[0])
            if problem.isGoalState(current_node[0]):    
                return current_node[1]
            enqueue(fringe, current_node, problem.getSuccessors(current_node[0]))


def enqueue(fringe, current_node, successors):
    actions = {
        "West": Directions.WEST,
        "East": Directions.EAST,
        "South": Directions.SOUTH,
        "North": Directions.NORTH 
    }
    for successor in successors:
        coordinates = successor[0]
        pathDirections = None
        pathDirections = current_node[1] + [actions[successor[1]]] if type(current_node[1]) is list else [actions[successor[1]]]
        cost = successor[2] + current_node[2] if len(current_node) > 2 else successor[2]
        fringe.push((coordinates, pathDirections, cost))
    return fringe

def depthFirstSearch(problem):
    return graph_search(util.Stack(), problem)

def breadthFirstSearch(problem):
    return graph_search(util.Queue(), problem)

def uniformCostSearch(problem):
    return graph_search(util.PriorityQueueWithFunction(ucsPriorityQueueFn), problem)

def ucsPriorityQueueFn(node):
    return node[2]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
# pathDirections = current_node[1].append(successor_action) if type(current_node[1]) is list else [actions[successor[1]]]

def aStarSearch(problem, heuristic=nullHeuristic):
    return graph_search(util.PriorityQueueWithFunction(createAStarQueueFunction(heuristic, problem)), problem)

def createAStarQueueFunction(heuristic, problem):
    return lambda node: heuristic(node[0], problem) + node[2]

    

ucs=uniformCostSearch
astar=aStarSearch
bfs=breadthFirstSearch
dfs=depthFirstSearch
