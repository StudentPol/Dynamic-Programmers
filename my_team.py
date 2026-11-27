# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


# Change: Search node (search.py)
class SearchNode:
    def __init__(self, parent, node_info):
        self.state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # Change: Compute the boundary (midline) based on the walls of the maze
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        home_x = mid_x - 1 if self.red else mid_x
        self.boundary = [(home_x, y) for y in range(walls.height) if not walls[home_x][y]]

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    # Change: visible enemy ghosts that can eat us (not scared)
    def get_dangerous_ghosts(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [a.get_position() for a in enemies
                if not a.is_pacman and a.get_position() is not None and a.scared_timer <= 2]

    # Change: A* (search). Cells next to a ghost cost more, so the
    # path avoids them. Returns the first action of the path (or None).
    def a_star_first_action(self, game_state, goals):
        walls = game_state.get_walls()
        ghosts = self.get_dangerous_ghosts(game_state)
        my_pos = game_state.get_agent_position(self.index)

        def heuristic(pos):
            return min([self.get_maze_distance(pos, g) for g in goals])

        def step_cost(pos):
            cost = 1
            for ghost in ghosts:
                d = self.get_maze_distance(pos, ghost)
                if d <= 1:
                    cost += 100
                elif d == 2:
                    cost += 10
            return cost

        start_node = SearchNode(None, (my_pos, None, 0))
        fringe = util.PriorityQueue()
        fringe.push(start_node, heuristic(my_pos))
        best_cost = {my_pos: 0}
        explored = set()

        while not fringe.is_empty():
            node = fringe.pop()
            if node.cost != best_cost[node.state] or node.state in explored:
                continue
            explored.add(node.state)

            if node.state in goals:
                path = node.get_path()
                return path[0] if len(path) > 0 else None

            x, y = node.state
            for action, (dx, dy) in [(Directions.NORTH, (0, 1)), (Directions.SOUTH, (0, -1)),
                                     (Directions.EAST, (1, 0)), (Directions.WEST, (-1, 0))]:
                successor = (x + dx, y + dy)
                if walls[successor[0]][successor[1]] or successor in explored:
                    continue
                successor_node = SearchNode(node, (successor, action, step_cost(successor)))
                if successor_node.cost < best_cost.get(successor, float('inf')):
                    best_cost[successor] = successor_node.cost
                    fringe.push(successor_node, successor_node.cost + heuristic(successor))
        return None


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying
        food_left = len(self.get_food(game_state).as_list())

        if carrying > 0:
            dist_home = min([self.get_maze_distance(my_pos, b) for b in self.boundary])
            ghosts = self.get_dangerous_ghosts(game_state)
            ghost_dist = min([self.get_maze_distance(my_pos, g) for g in ghosts]) if ghosts else None
            moves_left = game_state.data.timeleft // 4

            low_time = moves_left <= dist_home + 2
            immediate_danger = ghost_dist is not None and ghost_dist <= max(2, min(5, carrying))
            carrying_enough = carrying >= 5 + dist_home // 2
            if food_left <= 2 or low_time or immediate_danger or carrying_enough:
                action = self.a_star_first_action(game_state, self.boundary)
                if action is not None:
                    return action

        if food_left <= 2:
            actions = game_state.get_legal_actions(self.index)
            values = [self.evaluate(game_state, action) for action in actions]
            max_value = max(values)
            best_actions = [action for action, value in zip(actions, values) if value == max_value]
            return random.choice(best_actions)

        return ReflexCaptureAgent.choose_action(self, game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Change: keep away from ghosts that can eat us (multiagent)
        my_pos = successor.get_agent_state(self.index).get_position()
        ghosts = self.get_dangerous_ghosts(game_state)
        if my_pos == self.start:
            features['eaten'] = 1
        elif successor.get_agent_state(self.index).is_pacman and len(ghosts) > 0:
            ghost_dist = min([self.get_maze_distance(my_pos, g) for g in ghosts])
            if ghost_dist <= 1:
                features['ghost_too_close'] = 1
            if ghost_dist <= 5:
                features['ghost_danger'] = 6 - ghost_dist  # bigger when the ghost is closer

                # Change: when a ghost is close, a capsule makes it harmless
                capsules = self.get_capsules(successor)
                features['capsules_left'] = len(capsules)
                if len(capsules) > 0:
                    features['distance_to_capsule'] = min([self.get_maze_distance(my_pos, c) for c in capsules])

        # Change: do not stand still
        if action == Directions.STOP: features['stop'] = 1
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1,
                'eaten': -1000, 'ghost_too_close': -500, 'ghost_danger': -10,
                'capsules_left': -200, 'distance_to_capsule': -2, 'stop': -100}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    # Change: decide where to go when no invader is visible
    def choose_action(self, game_state):
        # If some of our food disappeared, an invader is there: go to that spot
        previous = self.get_previous_observation()
        if previous is not None:
            before = self.get_food_you_are_defending(previous).as_list()
            now = self.get_food_you_are_defending(game_state).as_list()
            eaten = [food for food in before if food not in now]
            if len(eaten) > 0:
                self.target = eaten[0]

        # Otherwise patrol the middle of our border
        my_pos = game_state.get_agent_position(self.index)
        if getattr(self, 'target', None) is None or my_pos == self.target:
            self.target = self.boundary[len(self.boundary) // 2]

        return ReflexCaptureAgent.choose_action(self, game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # Change: no invader in sight, go to the target (eaten food or border)
            features['target_distance'] = self.get_maze_distance(my_pos, self.target)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2,
                'target_distance': -2}
