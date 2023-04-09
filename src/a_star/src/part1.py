#!/usr/bin/env python3
import time
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.path as plt_path
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation, writers
import math
from heapq import heappush, heappop
from collections import defaultdict, OrderedDict
import sys
import argparse


START = [0.8, 1.1, 0]
GOAL = [5, 1]
RPM = [10, 20]
CLEARANCE = 0.1

def parse_args():
    parser = argparse.ArgumentParser(description="Solve for an best path via A*.") 
    parser.add_argument("-s", "--start", 
        default=START, nargs='+', type=float, help="Starting cell indices.")
    parser.add_argument("-g", "--goal", 
        default=GOAL, nargs='+', type=float, help="Goal cell indices.")
    parser.add_argument("-r", "--RPM", 
        default=RPM, nargs='+', type=float, help="Input RPM")
    parser.add_argument("-c", "--clearance", 
        default=CLEARANCE, type=float, help="hurdle avoidance clearance.")
    parser.add_argument("-v", "--visualize", action="store_true",
        help="Generate a video of the path planning process.")
    args,_ = parser.parse_known_args()
    args.start[2]*=180/3.14159
    choice = Choice(args.start, args.goal, args.RPM, args.clearance, args.visualize)
    return choice


class Matrix:
    def __init__(self, env_map, start_cell, counter=0):
        self.env_map = env_map
        self.counter = counter
        self.max_cells = env_map.dimension()[0]/start_cell.offset[0]\
                         * env_map.dimension()[1]/start_cell.offset[1]\
                         * 360/start_cell.offset[2]
        st = time.time()
        self.cells, self.hm_store = self.build(start_cell)
        print("Time taken to build matrix={:.3f}s ".format(time.time()-st))

    def build(self, start_cell):
        hm_store = defaultdict(dict)
        cells={}
        cell_hashes=set()
        current_cells = [start_cell] 
        while len(current_cells) != 0:
            new_cells = []
            for cell in current_cells:
                cell_hash = hash(cell)
                if cell_hash in cell_hashes:
                    continue
                if not self.env_map.check_valid(cell.vertices, self.counter):
                    continue
                if cell.previous_cell:
                    hm_store[hash(cell.previous_cell)][cell_hash] = cell.previous_price - cell.previous_cell.previous_price
                cells[cell_hash] = cell
                cell_hashes.add(cell_hash)
                next_cells = cell.get_next_cells()
                new_cells += next_cells
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%% (worst case: %d/%d)" % ('='*int(20*len(cells)/self.max_cells), int(100*len(cells)/self.max_cells),len(cells),self.max_cells))
            sys.stdout.flush()
            current_cells = new_cells
        print()
        return cells, hm_store

class Map:
    def __init__(self, joint1, joint2):
        self.joint1 = np.array(joint1)
        self.joint2 = np.array(joint2)
        self.limit1 = [self.joint1[0], self.joint2[0]]
        self.limit2 = [self.joint1[1], self.joint2[1]]
        self.workspace = plt_path.Path(np.array([
            joint1, 
            [joint2[0],joint1[1]],
            joint2,
            [joint1[0],joint2[1]],
            joint1
        ]))

    def dimension(self):
        return [self.limit1[1]-self.limit1[0],self.limit2[1]-self.limit2[0]]

    def check_workspace(self, point, counter=0):
        return self.workspace.contains_point(point, radius=-(counter-0.01))

    def check_hurdle(self, point, counter=0):
        return any([hurdle.inside(point,counter) for hurdle in self.hurdles])
    
    def check_valid(self, point, counter=0):
        xy = point[:2]
        return self.check_workspace(xy, counter) and not self.check_hurdle(xy, counter)

    def plot(self):
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        map_vertices = self.workspace.vertices
        plt.plot(map_vertices[:,0],map_vertices[:,1],'k')
        for hurdle in self.hurdles:
            hurdle.plot(ax)
        return fig,ax

class NewMap(Map):
    def __init__(self):
        super().__init__([0,0],[6,2])
        self.hurdles = [
            Polygon([[1.5,0.75],[1.65,0.75],[1.65,2],[1.5,2]]),
            Polygon([[2.5,0],[2.65,0],[2.65,1.25],[2.5,1.25]]),
            Circle((4,1.1),0.5)
        ]

class Movement:
    def __init__(self, RPM=[1,1], r=0.5, L=0.5, alpha=0.1):
        self.alpha = alpha
        self.r = r
        self.L = L
        self.movements = [
        [0, RPM[0]],
        [RPM[0], 0],
        [RPM[0], RPM[0]],
        [0, RPM[1]],
        [RPM[1], 0],
        [RPM[1], RPM[1]],
        [RPM[0], RPM[1]],
        [RPM[1], RPM[0]]
        ]

    def calc_offset(self, start_theta):
        """Calculate necessary X/Y/Theta resolution
        """
        _,moves,_ = self.attempt([0,0,0])
        X = set([abs(m[0]) for m in moves if m[0]!=0])
        Y = set([abs(m[1]) for m in moves if m[1]!=0])

        T = [m[2] for m in moves]
        T = set([m for m in T+[start_theta] if m!=0])

        # get recommended resolution:
        leading_digit = lambda x: -int(math.floor(math.log10(abs(x))))
        round_to_first = lambda x,d: np.floor(x*10**d)/10.0**d
        res_x = min([round_to_first(v,leading_digit(v)) for v in X])
        res_y = min([round_to_first(v,leading_digit(v)) for v in Y]) 
        res_t = min([round_to_first(v,leading_digit(v)) for v in T]) 
        print(res_x,res_y,res_t)

        # set some more reasonable limits
        res_x = max(res_x, 0.1)
        res_y = max(res_y, 0.1)
        res_t = max(res_t, 1)
        print(res_x,res_y,res_t)
        return [0.03, 0.03, 15]

    def attempt(self, current_position):
        movements = []
        attempts = []
        prices = []
        for movement in self.movements:
            current_angle = current_position[2]*np.pi/180
            new_alpha = self.r*(movement[1]-movement[0])*self.alpha/self.L
            dx = 0.5*self.r*(movement[0]+movement[1])*np.cos(new_alpha+current_angle)*self.alpha
            dy = 0.5*self.r*(movement[0]+movement[1])*np.sin(new_alpha+current_angle)*self.alpha
            new_x = current_position[0] + dx
            new_y = current_position[1] + dy
            new_angle = (current_position[2] + new_alpha*180/np.pi)%360
            movements.append(movement)
            attempts.append((new_x, new_y, new_angle))
            prices.append(self.alpha)
        return movements, attempts, prices

class Cell:
    offset = None
    hash_offset = None
    movement = None
    def __init__(self, vertices, previous_price=0, previous_cell=None, previous_cell_movement=None):
        self.vertices = np.array(vertices)
        self.new_vertices = self.round(vertices)
        self.previous_price = previous_price
        self.previous_cell = previous_cell
        self.previous_cell_movement = previous_cell_movement

    def __hash__(self):
        return hash(tuple(self.new_vertices+self.hash_offset))

    def __str__(self):
        return str(self.new_vertices)
    
    def __eq__(self, rhs):
        return self.new_vertices == rhs.new_vertices

    @classmethod
    def set_hash_offset(cls, offset):
        cls.hash_offset = np.array(offset)

    @classmethod
    def set_movement(cls, movement):
        cls.movement = movement

    @classmethod
    def set_offset(cls, offset):
        cls.offset = offset

    def round(self, vertices):
        new_value = [] 
        for value, result in zip(vertices, self.offset):
            new_value.append(round(value/result)*result)
        return new_value

    def next_price(self, target_cell):
        return np.linalg.norm(self.vertices[:2]-target_cell.vertices[:2])

    def is_goal(self, goal_cell, tolerance):
        return self.next_price(goal_cell) < tolerance 

    def get_next_cells(self):
        next_cells = []
        movements,attempts,prices = self.movement.attempt(self.vertices)
        for movement, move, price in zip(movements,attempts,prices): 
            next_cell = Cell(np.array(move), self.previous_price + price, self, movement)
            if (not self.previous_cell) or next_cell != self.previous_cell:
                next_cells.append(next_cell)
        return next_cells

class Hurdle(object):
    def __init__(self):
        pass

    def __contains__(self, val):
        return self.inside(val)

    @abstractmethod
    def inside(self,point,counter):
        pass
    
    @abstractmethod
    def plot(self):
        pass

class Polygon(Hurdle):
    def __init__(self, points):
        super(Polygon, self).__init__()
        self.points = plt_path.Path(np.array(points))

    def inside(self, point, counter=0):
        return self.points.contains_point(point, radius=(counter+0.1))

    def plot(self, ax):
        plt = patches.Polygon(xy=self.points.vertices)
        ax.add_artist(plt)
        plt.set_facecolor('k')

class Ellipse(Hurdle):
    def __init__(self, middle, major, minor):
        super(Ellipse,self).__init__()
        self.middle = middle
        self.major = major
        self.minor = minor

    def inside(self, point, counter=0):
        value = ((point[0]-self.middle[0])/(self.major/2+counter))**2.0 + ((point[1]-self.middle[1])/(self.minor/2+counter))**2.0
        return value <= 1

    def plot(self, ax):
        plt = patches.Ellipse(xy=self.middle,width=self.major,height=self.minor)
        ax.add_artist(plt)
        plt.set_facecolor('k')

class Circle(Ellipse):
    def __init__(self, middle, radius):
        super(Circle,self).__init__(middle, radius, radius)

class Choice:
    start               = None      
    goal                = None     
    RPM                 = None     
    clearance           = None         
    radius              = 0.120 
    wheel_radius        = 0.033   
    separation    = 0.178      
    time            = 3.0/60    
    def __init__(self, start, goal, RPM, clearance, visualize):
        self.start = start
        self.goal = goal
        self.RPM = RPM
        self.clearance = clearance
        self.visualize = visualize

class AStar:
    def __init__(self, matrix, source):
        self.matrix = matrix
        self.source = source
        self.source_hash = hash(source)
        self.distance = None
        self.previous= None
        self.goal = None

    def solve(self, goal_cell, goal_tolerance):
        st = time.time()
        queue = []
        distance = defaultdict(lambda: np.inf)
        previous = defaultdict(lambda: None)
        visited = OrderedDict()
        visited_hashes = set()
        distance[self.source_hash] = 0
        for cell_hash in self.matrix.cells.keys():
            heappush(queue, (distance[cell_hash], cell_hash))
        while len(queue) != 0:
            u_price,u = heappop(queue)
            if u in visited_hashes:
                continue
            visited[u] = u_price
            visited_hashes.add(u)
            if self.matrix.cells[u].is_goal(goal_cell, goal_tolerance):
                self.goal = u
                break
            for v,v_price in self.matrix.hm_store[u].items():
                if v in visited_hashes:
                    continue
                new_price = distance[u] + v_price 
                if new_price < distance[v]:
                    distance[v] = new_price 
                    previous[v] = u
                    + self.matrix.cells[v].next_price(goal_cell)
                    total_price = distance[v] + self.matrix.cells[v].next_price(goal_cell)
                    heappush(queue,(total_price,v))
        self.distance = distance
        self.previous= previous
        self.visited = visited
        print("Time taken to explore matrix ={:.3f}s ".format(time.time()-st))
        if not self.goal:
            print("Failed to find goal cell.")
            return False
        return True
    
    def path_details(self):
        current_hash = self.goal
        path = []
        price = self.distance[current_hash]
        while current_hash in self.previous.keys():
            path.append(current_hash)
            current_hash = self.previous[current_hash]
        path.append(self.source_hash)
        path.reverse()
        path = [self.matrix.cells[p] for p in path] 
        return path,price 
    
    def exploration_details(self, stop=False, destination=None):
        explored_cells = []
        explored_prices = []
        for index, cost in self.visited.items():
            explored_cells.append(self.matrix.cells[index])
            explored_prices.append(cost)
            if stop and self.matrix.cells[index]==destination:
                break
        return explored_cells, explored_prices

    def visited_details(self, stop=False, destination=None):
        visited_cells = []
        visited_prices = []
        for index, price in self.visited.items():
            visited_cells.append(self.matrix.cells[index])
            visited_prices.append(price)
            if stop and self.matrix.cells[index]==destination:
                break
        return visited_cells, visited_prices

def plot_path(env_map, best_path, display=False):
    env_map.plot()
    plt.plot(best_path[0].vertices[0], best_path[0].vertices[1], '*b')
    plt.text(best_path[0].vertices[0], best_path[0].vertices[1], 'START')
    plt.plot(best_path[-1].vertices[0], best_path[-1].vertices[1], '*r')
    plt.text(best_path[-1].vertices[0], best_path[-1].vertices[1], 'GOAL')
    X = [n.vertices[0] for n in best_path]
    Y = [n.vertices[1] for n in best_path]
    plt.plot(X,Y)
    if display:
        plt.show()

class ExplorationVisualizer:
    def __init__(self, env_map, best, cells, prices):
        self.env_map = env_map
        self.cells = cells
        self.prices = prices
        self.best = best
        self.fig,self.ax = self.env_map.plot()
        self.data1 = [self.cells[i].vertices[0] for i in range(len(self.cells)-1)]
        self.data2 = [self.cells[i].vertices[1] for i in range(len(self.cells)-1)]
        self.velocity1 = [0]*len(self.data1)
        self.velocity2 = [0]*len(self.data1)
        self.line = self.ax.quiver(
            self.data1,
            self.data2,
            self.velocity1,
            self.velocity2, 
            color='b', 
            angles='xy',
            scale=10)
        self.max_size = 500

    def plot(self, save=True):
        ani = FuncAnimation(
                self.fig, 
                self.new_path,
                interval=1,
                repeat=False,
                repeat_delay=10,
                init_func=self._init,
                frames=range(len(self.cells)),
                blit=False)
        plt.show()
   
    def _init(self):
        plt.plot(self.cells[0].vertices[0], self.cells[0].vertices[1], '*b')
        plt.text(self.cells[0].vertices[0], self.cells[0].vertices[1], 'START')
        plt.plot(self.best[-1].vertices[0], self.best[-1].vertices[1], '*r')
        plt.text(self.best[-1].vertices[0], self.best[-1].vertices[1], 'GOAL')
        return self.line,

    def new_path(self,frame):
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*int(20*frame/len(self.cells)), int(100*frame/len(self.cells))))
        sys.stdout.flush()
        if frame == len(self.cells)-1:
            X = [n.vertices[0] for n in self.best]
            Y = [n.vertices[1] for n in self.best]
            self.line, = plt.plot(X,Y)
            return self.line,
        self.velocity1[frame] = self.cells[frame+1].vertices[0]-self.cells[frame].vertices[0]
        self.velocity2[frame] = self.cells[frame+1].vertices[1]-self.cells[frame].vertices[1]
        self.line.set_UVC(self.velocity1, self.velocity2)
        return self.line,

if __name__ == "__main__":
    choice = parse_args()
    sleep = choice.time*60
    hurdle_map = NewMap()
    movement = Movement(
            choice.RPM,
            choice.wheel_radius,
            choice.separation,
            choice.time)
    offset = movement.calc_offset(choice.start[2])
    Cell.set_movement(movement)
    Cell.set_offset(offset)
    Cell.set_hash_offset(hurdle_map.dimension()+[0])
    counter = choice.radius + choice.clearance
    start_cell = Cell(choice.start)
    goal_cell = Cell(choice.goal+[0])
    print("Start cell: {}".format(start_cell))
    print("Goal  cell: {}".format(goal_cell))
    print("Generating matrix...")
    matrix = Matrix(hurdle_map, start_cell, counter=counter)
    print("Let's do A* search...")
    d = AStar(matrix, start_cell)
    if not d.solve(goal_cell, goal_tolerance=5*offset[0]):
        print("Unable to find path to the goal cell.")
        sys.exit(1)
    best_path,_ = d.path_details()
    plot_path(hurdle_map, best_path, True)
    if choice.visualize:
        visualizer = ExplorationVisualizer(
                hurdle_map, 
                best_path,
                *d.exploration_details(True, goal_cell)
        ) 
        visualizer.plot(True)
    else:
        plot_path(hurdle_map, best_path, True)
