import CompBolt as cbt
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Polygon
from shapely.ops import unary_union

def cost_function(x):
    A=cbt.Specimen(phi=[math.radians(x[0]), math.radians(x[1]), math.radians(-x[1]), math.radians(-x[0])], f=0.4)
    A.calculate_abd()
    A.ZandU_1984AM()
    A.yamada_sun()
    
    result = 0
    for k in range(len(x)):
        result_a = np.amax(A.failure_equation[k])
        if result_a > result:
            result = result_a
    return result

def fuzzify(inp, shape_vals):
    if inp < shape_vals[0]:
        f_left_triangle=1
    elif shape_vals[0] < inp < shape_vals[1]:
        f_left_triangle=(shape_vals[1]-inp)/(shape_vals[1]-shape_vals[0])
    else:
        f_left_triangle=0

    if inp < shape_vals[2]:
        f_triangle=0
    elif shape_vals[2] < inp <= (shape_vals[3]+shape_vals[2])/2:
        f_triangle=2*(inp-shape_vals[2])/(shape_vals[3]-shape_vals[2])    
    elif (shape_vals[3]+shape_vals[2])/2 < inp < shape_vals[3]:
        f_triangle=2*(shape_vals[3]-inp)/(shape_vals[3]-shape_vals[2])
    else:
        f_triangle=0

    if inp < shape_vals[4]:
        f_right_triangle=0
    elif shape_vals[4] < inp < shape_vals[5]:
        f_right_triangle=(inp-shape_vals[4])/(shape_vals[5]-shape_vals[4])
    else:
        f_right_triangle=1
    return [f_left_triangle, f_triangle, f_right_triangle]

def operate_implicate(type, fuzzy_NCBPE, fuzzy_weight, shape_vals):
    if type == 'left':
        coords=[(shape_vals[0], 0.0), 
                (shape_vals[0], max(fuzzy_NCBPE, fuzzy_weight)), 
                (-max(fuzzy_NCBPE, fuzzy_weight)*(shape_vals[1]-(shape_vals[0]))+shape_vals[1], max(fuzzy_NCBPE, fuzzy_weight)),
                (shape_vals[1], 0.0)]

    elif type == 'mid':
        coords=[(shape_vals[2], 0.0), 
                ((max(fuzzy_NCBPE, fuzzy_weight)*(shape_vals[3]-shape_vals[2])/2)+shape_vals[2], max(fuzzy_NCBPE, fuzzy_weight)),
                (-(max(fuzzy_NCBPE, fuzzy_weight)*(shape_vals[3]-shape_vals[2])/2)+shape_vals[3], max(fuzzy_NCBPE, fuzzy_weight)),
                (shape_vals[3], 0.0)]                

    elif type == 'right':
        coords=[(shape_vals[4], 0.0), 
                (max(fuzzy_NCBPE, fuzzy_weight)*(shape_vals[5]-shape_vals[4])+shape_vals[4], max(fuzzy_NCBPE, fuzzy_weight)), 
                (shape_vals[5], max(fuzzy_NCBPE, fuzzy_weight)), 
                (shape_vals[5], 0.0)]       
    return coords

def aggregate_defuzzify(coords_a, coords_b, coords_c):
    active_polygons=[]
    if Polygon(coords_a).area > 0.0:
        active_polygons.append(Polygon(coords_a))
    if Polygon(coords_b).area > 0.0:
        active_polygons.append(Polygon(coords_b))
    if Polygon(coords_c).area > 0.0:
        active_polygons.append(Polygon(coords_c))
    x = unary_union(active_polygons)
    return x.centroid.x

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.weight_i=1.1           # weight individual
        self.ncbpe_i=[]             # normalized current best performance evaluation individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update particle weight
    def update_weight(self):
        # normalize performance metric
        self.ncbpe_i=(self.err_i-2.0)/(8.0-2.0)

        # set fuzzy membership parameters
        a=[0, 0.45, 0.35, 0.75, 0.5, 1.0]
        b=[0.2, 0.6, 0.4, 0.9, 0.6, 1.1]
        c=[-0.12, -0.02, -0.04, 0.04, 0, 0.05]

        # fuzzify normalized performance metric and weight 
        fuzzy_NCBPE=fuzzify(self.ncbpe_i, a)
        fuzzy_weight=fuzzify(self.weight_i, b)

        # apply fuzzy operation (OR = max) and implication method (min)
        trunc_left_triangle=operate_implicate('left', fuzzy_NCBPE[0], fuzzy_weight[0], c)
        trunc_triangle=operate_implicate('mid', fuzzy_NCBPE[1], fuzzy_weight[1], c)
        trunc_right_triangle=operate_implicate('right', fuzzy_NCBPE[2], fuzzy_weight[2], c)
        
        # apply aggregation method (max) and defuzzify (centroid) 
        weight_change=aggregate_defuzzify(trunc_left_triangle, trunc_triangle, trunc_right_triangle)
        # print(weight_change)

        # update weight value
        self.weight_i=self.weight_i+weight_change
        if self.weight_i > 1.1:
            self.weight_i = 1.1
        if self.weight_i < 0.2:
            self.weight_i = 0.2
        print(self.weight_i)

    # update particle velocity
    def update_velocity(self,pos_best_g):
        c1=2        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=self.weight_i*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   
        pos_best_g=[]                   
        list_best_g=[]

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        y1 = []
        x = []
        x_plot = []
        example_weight = []
        
        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update the weights, velocities and positions
            for j in range(0,num_particles):
                swarm[j].update_weight()
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
                y1.append(pos_best_g)
                x.append(i)
            
            i+=1            
            x_plot.append(i)    
            example_weight.append(swarm[0].weight_i)
            list_best_g.append(err_best_g)
            
            fig, axs = plt.subplots(1,3)
            axs[0].plot(x_plot, example_weight)
            axs[1].plot(x_plot, list_best_g)
            axs[2].plot(x, y1)
            plt.show()

if __name__ == "__PSO__":
    main()

initial=[60,40]              
bounds=[(-90,90),(-90,90)]
PSO(cost_function,initial,bounds,num_particles=15,maxiter=100)
