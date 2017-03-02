# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		foodlist = newFood.asList()
		nonMovePenalty = -10 if successorGameState.getPacmanPosition() == currentGameState.getPacmanPosition() else 0
		closestGhost = distanceToClosestGhost(newPos, newGhostStates, successorGameState)
		# print newScaredTimes[0]
		modGhostScore =  -(closestGhost * newScaredTimes[0]) if newScaredTimes[0] > 0 else closestGhost
		# print modGhostScore
		positionScore =  closestGhost + successorGameState.getScore() + nonMovePenalty - distanceToClosestFood(newPos, foodlist) 
		return positionScore

# def manahtanDistance(p1, p2):
# 	return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# def distanceToClosestGhost(pPos, ghostStates, gameState):
# 	minGhostDistance = 1000
# 	for ghost in ghostStates:
# 		ghostDistance = manahtanDistance(pPos, ghost.configuration.pos)
# 		if minGhostDistance > ghostDistance:
# 			minGhostDistance = ghostDistance
# 	return minGhostDistance

# def distanceToClosestFood(pPos, allFood):
# 	closestFood = 0
# 	if len(allFood) > 0:
# 		closestFood = manahtanDistance(pPos, min(allFood, key=lambda foodPosition: manahtanDistance(pPos, foodPosition)))
# 	return closestFood 



def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', f=.2, p=.2, g=.3, s=.4):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)
		self.totalNodesExplored = 0



def MiniMaxBase(state, currentDepth, maxDepth, evalFunc, oppentDicisionMaking, prune=False, accurateDistance=False):
	def minVal():
		actionScores = [MiniMaxBase(state.generateSuccessor(agentIndex, action), currentDepth, maxDepth, evalFunc, oppentDicisionMaking, prune=prune, accurateDistance=accurateDistance) for action in state.getLegalActions(agentIndex)]
		return oppentDicisionMaking(sorted(actionScores))

	def maxVal():
		actionScores = [MiniMaxBase(state.generateSuccessor(agentIndex, action), currentDepth, maxDepth, evalFunc, oppentDicisionMaking, prune=prune, accurateDistance=accurateDistance) for action in state.getLegalActions(agentIndex)]
		return sorted(actionScores).pop(-1)

	agentIndex = currentDepth % state.getNumAgents()
	if state.data._win or state.data._lose or currentDepth > (maxDepth * state.getNumAgents()):
		return evalFunc(state)
	elif agentIndex > 0:
		currentDepth += 1
		return minVal()
	else:
		currentDepth += 1
		return maxVal()
	

class MinimaxAgent(MultiAgentSearchAgent):
	def bestDecision(self, sortedScores):
		return sortedScores.pop(0)

	def getAction(self, gameState):
		sortedActions = sorted(gameState.getLegalActions(self.index), key=lambda action: MiniMaxBase(gameState.generateSuccessor(0, action), 0, self.depth, betterEvaluationFunction, self.bestDecision))
		return sortedActions.pop(-1)

class ExpectimaxAgent(MultiAgentSearchAgent):
	def randomDecision(self, sortedScores):
		return sortedScores.pop(0)

	def getAction(self, gameState):
		sortedActions = sorted(gameState.getLegalActions(self.index), key=lambda action: MiniMaxBase(gameState.generateSuccessor(0, action), 0, self.depth, betterEvaluationFunction, self.randomDecision))
		return sortedActions.pop(-1)




def betterEvaluationFunction(gameState, acurate=False):
	current_position = gameState.getPacmanPosition()
	coor_match = {
		"North": [0, 1],
		"East": [1, 0],
		"South": [0, -1],
		"West": [-1, 0],
		"Stop": [0, 0]
	}
	
	def getLegalActions(position):
		legal_actions = []
		for action in ["North", "East", "South", "West", "Stop"]:
			next_pos = map(sum, zip(position, coor_match[action]))
			trans = (next_pos[0] - 1) * gameState.data.layout.height + next_pos[1]
			if gameState.data.layout.walls[trans]:
				legal_actions.append(action)
		return  legal_actions

	def manahtan_distance_to(items_to_find):
			if not items_to_find: return 1
			return min(abs(current_position[0] - item_position[0]) + abs(current_position[1] - item_position[1]) for item_position in items_to_find)
	def actual_distance_to(items_to_find):
		fringe = list()
		explored = list()
		fringe.extend([ [[action], map(sum, zip(current_position, coor_match[action]))] for action in gameState.getLegalActions(0) ]) # creates node format of [[direction], [coordinates]]
		while fringe:
			current_node = fringe.pop(0)
			if current_node[1] not in explored:
				if current_node[1] in items_to_find:
					return len(current_node[0])
				explored.append(current_node[1])
				fringe.extend([ [current_node.append(direction), map(sum, zip(current_node[1], coor_match[direction]))] for direction in getLegalActions(current_node[1]) ])
		return 1
	distance_to = actual_distance_to if acurate else manahtan_distance_to 
	ghost_states = gameState.getGhostStates()

	dt_closest_food = distance_to(gameState.getFood().asList())
	dt_closest_ghost = distance_to([ghost.configuration.pos for ghost in ghost_states])
	dt_cloest_capsuel = distance_to(gameState.data.capsules)

	# dt_closest_food = 0
	# dt_closest_ghost = 0
	# dt_cloest_capsuel =	0

	if len(ghost_states) and ghost_states[0].scaredTimer > 0: dt_closest_ghost = -dt_closest_ghost


	if gameState.data._capsuleEaten: dt_cloest_capsuel = -dt_cloest_capsuel
	# killing_distance = 0 if 0 < dt_closest_ghost <= 2 else 1000
	score = (gameState.getScore() * .50 ) - (dt_closest_food * .20) - (dt_cloest_capsuel  * .10) - (dt_closest_ghost * .20)
	return score


class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""
	def getAction(self, gameState):
		bestAction = None
		bestActionScore = float("-inf")
		upStreamMin = float("inf")
		upStreamMax = float("-inf")
		for action in gameState.getLegalActions(self.index):
			actionScore = self.evaluateSuccessor(gameState.generateSuccessor(0, action), 0, upStreamMin, upStreamMax)
			if actionScore >= bestActionScore:
				bestAction = action
				bestActionScore = actionScore
		return bestAction

	def evaluateSuccessor(self, state, depth, upStreamMin, upStreamMax):
		agentIndex = depth % state.getNumAgents()
		if state.data._win or state.data._lose or depth >= (self.depth * state.getNumAgents()):
			return self.evaluationFunction(state)
		elif agentIndex > 0:
			return self.minVal(state, depth, agentIndex, upStreamMin, upStreamMax)
		else:
			return self.maxVal(state, depth, agentIndex, upStreamMin, upStreamMax)

	def minVal(self, state, depth, agentIndex, upStreamMin, upStreamMax):
		depth += 1
		actionScore = self.evaluateSuccessor(state.generateSuccessor(agentIndex, random.choice(state.getLegalActions(agentIndex))), depth, upStreamMin, upStreamMax)
		upStreamMin = min(actionScore, upStreamMin)
		return actionScore
		# minValue = float('inf')
		# for action in state.getLegalActions(agentIndex):
		# 	actionScore =  self.evaluateSuccessor(state.generateSuccessor(agentIndex, action), depth, upStreamMin, upStreamMax)
		# 	minValue = min(minValue, actionScore)
		# 	if minValue <= upStreamMax: break
		# 	upStreamMin = min(minValue, upStreamMin)
		# return minValue

	def maxVal(self, state, depth, agentIndex, upStreamMin, upStreamMax):
		depth += 1
		maxValue = float('-inf')
		for action in state.getLegalActions(agentIndex):
			actionScore = self.evaluateSuccessor(state.generateSuccessor(agentIndex, action), depth, upStreamMin, upStreamMax)
			maxValue = max(maxValue, actionScore)
			if upStreamMin <= maxValue: break
			upStreamMax = max(maxValue, upStreamMax)
		return maxValue

# Abbreviation
better = betterEvaluationFunction

