import pygame as pg
import random as rnd
import copy
import math
pg.init()

#Set Pygame
screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
run = True

width, height = screen.get_size()

Clock = pg.time.Clock()

surface_size = [640, 640]

surface = pg.Surface(surface_size)

side_size = 0
scale = 0

font = pg.font.Font(None, 22)

#Set game
r_specimen = 10
r_enemy = 150
dr_enemy = 150
speed = 2
n_specimens = 100
time = 0
mutate_size = 2
start_pos = [160, 320]

#Variables
Inputs = 6
Outputs = 2
specimens = []
winners = []
generation = 1

#Functions
def Sigmoid(x):
    x = max(-50, min(x, 50))
    y = 1/(1 + math.exp(-x))
    return y

def Distance2D(v1, v2):
    return ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)**(1/2)

def Direction1D(x1, x2):
    return x2 - x1

def Vector_unit(v1, v2):
    dis = Distance2D(v1, v2)
    if dis == 0:
        return [0, 0]
    return [(v1[0] - v2[0])/dis, (v1[1] - v2[1])/dis]

def Scale(v1, x):
    return [v1[0] * x, v1[1] * x]

#Classes
class Neuron:
    def __init__(self):
        self.weights = []
        self.bias = 0
    
    def Create(self, n_inputs):
        self.weights.clear()
        self.bias = rnd.uniform(-1, 1)
        for I in range(n_inputs):
            self.weights.append(rnd.uniform(-1, 1))
    
    def Output(self, v_inputs):
        sum_w = 0
        for w in range(len(self.weights)):
            sum_w += v_inputs[w] * self.weights[w]
        result = Sigmoid(sum_w + self.bias)
        return result
    
    def Mutate(self):
        for mutate_w in range(len(self.weights)):
            if (rnd.random() < 0.3):
                self.weights[mutate_w] += rnd.uniform(-mutate_size, mutate_size)
        if (rnd.random() < 0.5):
            self.bias += rnd.uniform(-mutate_size, mutate_size)

class Neural_layer:
    def __init__(self):
        self.neurons = []
    
    def Create(self, n_inputs):
        n_neurons = rnd.randint(2, 16)
        for N in range(n_neurons):
            self.neurons.append(Neuron())
            self.neurons[N].Create(n_inputs)
    
    def Output(self, v_inputs):
        v_output = []
        for id_neuron in range(len(self.neurons)):
            v_output.append(self.neurons[id_neuron].Output(v_inputs))
        return v_output
    
    def Mutate_neuron(self, n_neuron):
        #Actualizar
        for update_n in self.neurons:
            if (n_neuron > 0):
                update_n.weights.append(rnd.uniform(-1, 1))
            elif (n_neuron < 0):
                update_n.weights.pop()
        
        if (rnd.random() < 0.2):
            if (rnd.random() < 0.5) and (len(self.neurons) < 16):
                self.neurons.append(Neuron())
                self.neurons[len(self.neurons) - 1].Create(len(self.neurons[0].weights))
                return 1
            elif (len(self.neurons) > 2):
                self.neurons.pop()
                return -1
        return 0
    
    def Mutate(self):
        for mutate_w_b in self.neurons:
            mutate_w_b.Mutate()

class Neural_net:
    def __init__(self):
        self.hidden_layers = []
        self.output_layer = []
    
    def Create(self, n_inputs, n_outputs):
        n_layers = rnd.randint(1, 4)
        for L in range(n_layers):
            self.hidden_layers.append(Neural_layer())
            if (L == 0):
                self.hidden_layers[L].Create(n_inputs)
            else:
                self.hidden_layers[L].Create(len(self.hidden_layers[L - 1].neurons))
        for O in range(n_outputs):
            self.output_layer.append(Neuron())
            self.output_layer[O].Create(len(self.hidden_layers[n_layers - 1].neurons))
    
    def Output(self, v_inputs):
        inputs = v_inputs
        for id_layer in range(len(self.hidden_layers)):
            inputs = self.hidden_layers[id_layer].Output(inputs)
        
        v_output = []
        for id_output in range(len(self.output_layer)):
            v_output.append(self.output_layer[id_output].Output(inputs))
        return v_output
    
    def Mutate(self):
        #Mutar capas
        remove_c = 0
        add_c = 0
        if (rnd.random() < 0.01):
            if (rnd.random() < 0.5):
                if (len(self.hidden_layers) > 1):
                    remove_c = rnd.randint(1, len(self.hidden_layers))
                    self.hidden_layers.pop(remove_c - 1)
            else:
                if (len(self.hidden_layers) < 4):
                    add_c = rnd.randint(1, len(self.hidden_layers))
                    self.hidden_layers.insert(add_c - 1, Neural_layer())
                if (add_c == 1):
                    self.hidden_layers[add_c - 1].Create(Inputs)
                elif (add_c > 0):
                    self.hidden_layers[add_c - 1].Create(len(self.hidden_layers[add_c - 2].neurons))
            if (remove_c > 0):
                if (remove_c == 1):
                    for neuron in self.hidden_layers[remove_c - 1].neurons:
                        neuron.Create(Inputs)
                elif (remove_c == len(self.hidden_layers) + 1):
                    for neuron in self.output_layer:
                        neuron.Create(len(self.hidden_layers[remove_c - 2].neurons))
                else:
                    for neuron in self.hidden_layers[remove_c - 1].neurons:
                        neuron.Create(len(self.hidden_layers[remove_c - 2].neurons))
            if (add_c > 0):
                for neuron in self.hidden_layers[add_c].neurons:
                    neuron.Create(len(self.hidden_layers[add_c - 1].neurons))
        #Mutar numero de neuronas
        new_neuron = 0
        for layer_mutate in self.hidden_layers:
            new_neuron = layer_mutate.Mutate_neuron(new_neuron)
        for update_n in self.output_layer:
            if (new_neuron > 0):
                update_n.weights.append(0)
            elif (new_neuron < 0):
                update_n.weights.pop()
        #Mutar pesos y sesgos
        #Hidden layers
        for mutate_w_b in self.hidden_layers:
            mutate_w_b.Mutate()
        for mutate_w_b in self.output_layer:
            mutate_w_b.Mutate()
     
class Specimen:
    def __init__(self):
        self.brain = Neural_net()
        self.points = 0
        self.num_neurons = 0
        self.r = r_specimen
        self.color = "#00ff00"
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.bx = self.x
        self.by = self.y
        self.pos = [self.x, self.y]
        self.bpos = [self.bx, self.by]
    
    def Create(self, n_inputs, n_outputs):
        self.brain.Create(n_inputs, n_outputs)
        for layer in self.brain.hidden_layers:
            self.num_neurons += len(layer.neurons)
    
    def Move(self, v_inputs, dft):
        self.bx = self.x
        self.by = self.y
        self.bpos = [self.bx, self.by]
        df = self.brain.Output(v_inputs)
        df[0] = ((df[0] * 2) - 1)*speed
        df[1] = ((df[1] * 2) - 1)*speed
        self.x += max(-10, min(df[0], 10))*dft
        self.y += max(-10, min(df[1], 10))*dft
        self.pos = [self.x, self.y]
    
    def Set_pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.pos = [self.x, self.y]
    
    def Draw(self, surface):
        pg.draw.circle(surface, self.color, self.pos, self.r, 0)
    
    def Points(self):
        fitness = self.points - self.num_neurons
        return fitness
    
    def Mutate(self):
        s_copy = copy.deepcopy(self)
        s_copy.brain.Mutate()
        s_copy.points = 0
        s_copy.num_neurons = 0
        s_copy.x = start_pos[0]
        s_copy.y = start_pos[1]
        s_copy.bx = s_copy.x
        s_copy.by = s_copy.y
        s_copy.pos = [s_copy.x, s_copy.y]
        s_copy.bpos = [s_copy.bx, s_copy.by]
        for count_n in range(len(s_copy.brain.hidden_layers)):
            s_copy.num_neurons += len(s_copy.brain.hidden_layers[count_n].neurons)
        return s_copy
    
    def Copy(self):
        new_copy = copy.deepcopy(self)
        new_copy.points = 0
        new_copy.x = start_pos[0]
        new_copy.y = start_pos[1]
        new_copy.bx = new_copy.x
        new_copy.by = new_copy.y
        new_copy.pos = [new_copy.x, new_copy.y]
        new_copy.bpos = [new_copy.bx, new_copy.by]
        return new_copy
        

class Enemy:
    def __init__(self):
        self.r = r_enemy
        self.color = "#ff0000"
        self.x = surface_size[0]/2
        self.y = surface_size[1]/2
        self.dx = 0
        self.dy = 0
        self.pos = [self.x + self.dx, self.y + self.dy]
        self.angle = 0
        self.da = 1
        self.dr = dr_enemy
    
    def Move(self, dft):
        self.angle += self.da*dft
        if (self.angle >= 360) or (self.angle <= -360):
            self.angle = 0
        rad = math.radians(self.angle)
        self.dx = (self.dr) * math.cos(rad)
        self.dy = (self.dr) * math.sin(rad)
        self.pos = [self.x + self.dx, self.y + self.dy]
    
    def Draw(self, surface):
        pg.draw.circle(surface, self.color, (self.pos[0], self.pos[1]), self.r, 0)
    
    def Mutate(self):
        self.angle = 0
        #if (rnd.random() < 0.05):
            #self.da *= -1
        #if (rnd.random() < 0.3):
            #self.da = rnd.randint(1, 3)
        #if (rnd.random() < 0.3):
            #self.r = rnd.randint(40, 120)
        #if (rnd.random() < 0.3):
            #self.dr = rnd.randint(150, 200)

#Before
e1 = Enemy()
for specimen_id in range(n_specimens):
    specimens.append(Specimen())
    specimens[specimen_id].Create(Inputs, Outputs)
game_specimens = len(specimens)

#Game
while (run):
    if (round(Clock.get_fps()) >= 1):
        dftime = 60/round(Clock.get_fps())
    else:
        dftime = 1
    time += 1*dftime
    #Settings
    width, height = screen.get_size()
    if (width > height):
        side_size = height
    else:
        side_size = width
    
    #Events
    for event in pg.event.get():
        if event.type == pg.QUIT:
            run = False
    
    #Logic
    #Finish
    if time >= 1200:
        for s_id in specimens[:]:
            winners.append(specimens.pop())
    #Mutacion
    if (len(specimens) <= 0):
        generation += 1
        time = 0
        winners.sort(key=lambda specimen: specimen.Points(), reverse=True)
        for id_winner in range(10):
            n_copies = round(((10 - id_winner)*20)/11)
            for id_mutantes in range(n_copies):
                if (id_mutantes == 0):
                    specimens.append(winners[id_winner].Copy())
                else:
                    specimens.append(winners[id_winner].Mutate())
        
        e1.Mutate()
        game_specimens = len(specimens)
    
    #Seleccion
    for specimen in specimens:
        specimen_inputs = [Direction1D(specimen.pos[0], surface_size[0]/2)/surface_size[0], Direction1D(specimen.pos[1], surface_size[1]/2)/surface_size[1], Direction1D(specimen.pos[0], e1.pos[0])/(surface_size[0]/2 - 180), Direction1D(specimen.pos[1], e1.pos[1])/(surface_size[1]/2 - 180), Distance2D(specimen.pos, [surface_size[0]/2, surface_size[1]/2])/surface_size[0], Distance2D(specimen.pos, e1.pos)/surface_size[0]]
        specimen.Move(specimen_inputs, dftime)
        #specimen.points += round(Distance2D(specimen.pos, specimen.bpos))
    e1.Move(dftime)
    
    #Detectar colision
    for specimen in specimens[:]:
        specimen.points += 5
        if (((Distance2D(specimen.pos, Scale(surface_size, 1/2)) + r_specimen) > surface_size[0]/2 - 20)):
            new_pos = Scale(Vector_unit(specimen.pos, Scale(surface_size, 1/2)), surface_size[0]/2 - (r_specimen + 20))
            new_pos[0] += surface_size[0]/2
            new_pos[1] += surface_size[1]/2
            specimen.Set_pos(new_pos)
            winners.append(copy.deepcopy(specimen))
            specimens.remove(specimen)
            continue
        if (Distance2D(specimen.pos, e1.pos) < r_specimen + e1.r):
            winners.append(copy.deepcopy(specimen))
            specimens.remove(specimen)
            continue
    
    #Draw
    #Clean
    screen.fill("#000000")
    surface.fill("#000000")
    #Layer 1
    for specimen in specimens:
        specimen.Draw(surface)
    #Layer 2
    e1.Draw(surface)
    #Layer 3
    pg.draw.circle(surface, "#ffffff", (surface_size[0]/2, surface_size[1]/2), surface_size[0]/2 - 10, 10)
    #Layer 3
    text_1 = font.render(f"Generación: {generation}", True, "#ffffff")
    text_2 = font.render(f"Especimenes: {game_specimens}", True, "#ffffff")
    text_3 = font.render(f"Tiempo: {time//60}s", True, "#ffffff")
    text_4 = font.render(f"Vivos: {len(specimens)}", True, "#ffffff")
    text_5 = font.render(f"FPS: {round(Clock.get_fps())}", True, "#ffffff")
    
    surface.blit(text_1, (10, 10))
    surface.blit(text_2, (10, 30))
    surface.blit(text_3, (10, 50))
    surface.blit(text_4, (10, 70))
    surface.blit(text_5, (10, 90))
    #Scale
    scale = side_size/(surface_size[0])
    scale_s = [scale*surface_size[0], scale*surface_size[1]]
    
    scale_surface = pg.transform.scale(surface, (scale_s[0], scale_s[1]))
    #Draw surface
    screen.blit(scale_surface, ((width - scale_s[0])/2, (height - scale_s[1])/2))
    #Update
    pg.display.flip()
    Clock.tick(60)

pg.quit()