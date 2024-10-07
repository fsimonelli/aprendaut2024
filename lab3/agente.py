import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import numpy as np
import pygame

# Cuántos bins queremos por dimensión
# Pueden considerar variar este parámetro
bins_per_dim = 20

# Estado:
# (x, y, x_vel, y_vel, theta, theta_vel, pie_izq_en_contacto, pie_derecho_en_contacto)
NUM_BINS = [bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, bins_per_dim, 2, 2]
NUM_STATES = np.prod(NUM_BINS)

env = gym.make('LunarLander-v2')
env.reset()

# Tomamos los rangos del env
OBS_SPACE_HIGH = env.observation_space.high
OBS_SPACE_LOW = env.observation_space.low
OBS_SPACE_LOW[1] = 0 # Para la coordenada y (altura), no podemos ir más abajo que la zona dea aterrizae (que está en el 0, 0)

# Los bins para cada dimensión
bins = [
    np.linspace(OBS_SPACE_LOW[i], OBS_SPACE_HIGH[i], NUM_BINS[i] - 1)
    for i in range(len(NUM_BINS) - 2) # last two are binary
]
# Se recomienda observar los bins para entender su estructura
# print ("Bins: ", bins)

def discretize_state(state, bins):
    """Discretize the continuous state into a tuple of discrete indices."""
    state_disc = list()
    for i in range(len(state)):
        if i >= len(bins):  # For binary features (leg contacts)
            state_disc.append(int(state[i]))
        else:
            state_disc.append(
                np.digitize(state[i], bins[i])
            )
    return tuple(state_disc)

class Agente:
    def elegir_accion(self, estado, max_accion, explorar = True) -> int:
        """Elegir la accion a tomar en el estado actual y el espacio de acciones
            - estado_anterior: el estado desde que se empezó
            - estado_siguiente: el estado al que se llegó
            - accion: la acción que llevo al agente desde estado_anterior a estado_siguiente
            - recompensa: la recompensa recibida en la transicion
            - terminado: si el episodio terminó
        """
        pass

    def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
        """Aprender a partir de la tupla 
            - estado_anterior: el estado desde que se empezó
            - estado_siguiente: el estado al que se llegó
            - accion: la acción que llevo al agente desde estado_anterior a estado_siguiente
            - recompensa: la recompensa recibida en la transicion
            - terminado: si el episodio terminó en esta transición
        """
        pass

    def fin_episodio(self):
        """Actualizar estructuras al final de un episodio"""
        pass

class AgenteRL(Agente):
    # Pueden agregar parámetros al constructor
    def __init__(self, states, actions, gamma) -> None:
        super().__init__()
        self.Q = np.zeros((states,actions))
        self.gamma = gamma
        self.alpha = 0.3
        # Agregar código aqui
    
    def elegir_accion(self, estado, max_accion, explorar = True) -> int:
        discretized_state = discretize_state(estado,bins)
        row = discretized_state[0] * 12800000 + discretized_state[1] * 640000 + discretized_state[2] * 32000 + discretized_state[3] * 1600 + discretized_state[4] * 80 + discretized_state[5] * 4 + discretized_state[6] * 2 + discretized_state[7] 
        return np.argmax(self.Q[row])
    
    def aprender(self, estado_anterior, estado_siguiente, accion, recompensa, terminado):
        # Aprendizaje Q
        discretized_state_anterior = discretize_state(estado_anterior, bins)
        discretized_state_siguiente = discretize_state(estado_siguiente, bins)
        
        row_anterior = discretized_state_anterior[0] * 12800000 + discretized_state_anterior[1] * 640000 + discretized_state_anterior[2] * 32000 + discretized_state_anterior[3] * 1600 + discretized_state_anterior[4] * 80 + discretized_state_anterior[5] * 4 + discretized_state_anterior[6] * 2 + discretized_state_anterior[7]
        
        row_siguiente = discretized_state_siguiente[0] * 12800000 + discretized_state_siguiente[1] * 640000 + discretized_state_siguiente[2] * 32000 + discretized_state_siguiente[3] * 1600 + discretized_state_siguiente[4] * 80 + discretized_state_siguiente[5] * 4 + discretized_state_siguiente[6] * 2 + discretized_state_siguiente[7]
        
        # Fórmula de actualización de Q-learning
        if not terminado:
            self.Q[row_anterior][accion] += (1 - self.alpha) * self.Q[row_anterior][accion] + self.alpha * (recompensa + self.gamma * np.max(self.Q[row_siguiente]))
        else:
            # Si el episodio terminó, no hay valor de futuro
            self.Q[row_anterior][accion] += (1 - self.alpha) * self.Q[row_anterior][accion] + self.alpha * (recompensa)


    def fin_episodio(self):
        # Agregar código aqui
        pass
    
    
def ejecutar_episodio(agente, aprender = True, render = True, max_iteraciones=20000):
    entorno = gym.make('LunarLander-v2', render_mode='human').env
    
    iteraciones = 0
    recompensa_total = 0

    termino = False
    truncado = False
    estado_anterior, info = entorno.reset()
    #print(estado_anterior)
    #print(discretize_state(estado_anterior,bins))
    while iteraciones < max_iteraciones and not termino and not truncado:
        if render:
            entorno.render()
        # Le pedimos al agente que elija entre las posibles acciones (0..entorno.action_space.n)
        accion = agente.elegir_accion(estado_anterior, entorno.action_space.n, aprender)
        # Realizamos la accion
        estado_siguiente, recompensa, termino, truncado, info = entorno.step(accion)
        # Le informamos al agente para que aprenda
        if (aprender):
            agente.aprender(estado_anterior, estado_siguiente, accion, recompensa, termino)

        estado_anterior = estado_siguiente
        iteraciones += 1
        recompensa_total += recompensa
    if (aprender):
        print(recompensa_total)
        agente.fin_episodio()
    entorno.close()
    return recompensa_total

pygame.init()
entorno = gym.make('LunarLander-v2', render_mode='human').env
pygame.display.set_caption('Lunar Lander')
agente = AgenteRL(np.prod(NUM_BINS),entorno.action_space.n,1)
exitos = 0
recompensa_episodios = []
num_episodios = 00
for i in range(num_episodios):
    print("Episodio:",i)
    recompensa = ejecutar_episodio(agente)
    # Los episodios se consideran exitosos si se obutvo 200 o más de recompensa total
    if (recompensa >= 200):
        exitos += 1
    recompensa_episodios += [recompensa]
print(f"Tasa de éxito: {exitos / num_episodios}. Se obtuvo {np.mean(recompensa_episodios)} de recompensa, en promedio")

pygame.quit()