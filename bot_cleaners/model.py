from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np


class Celda(Agent):
    def __init__(self, unique_id, model, suciedad: bool = False):
        super().__init__(unique_id, model)
        self.sucia = suciedad


class Mueble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class RobotLimpieza(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sig_pos = None
        self.movimientos = 0
        self.carga = 100
        self.bloqueo = None  # Agrega el atributo de bloqueo
        self.cargasCompletadas = 0

    def recargar(self, cantidad):
        self.carga += cantidad
        if(self.carga > 100):
            self.carga == 100
    
    def buscar_suciedad_extendida(self, radio_busqueda=20):
        pos_actual = self.pos

        # Obtener las celdas en un radio de búsqueda alrededor del robot
        celdas_cercanas = list(self.model.grid.iter_neighborhood(pos_actual, moore=True, radius=radio_busqueda))

        # Filtrar las celdas que no están bloqueadas por otros robots
        celdas_disponibles = [celda for celda in celdas_cercanas if celda not in [r.bloqueo for r in self.model.schedule.agents if isinstance(r, RobotLimpieza)]]

        # Filtrar las celdas sucias
        celdas_sucias = [celda for celda in celdas_disponibles if any(isinstance(agente, Celda) and agente.sucia for agente in self.model.grid.get_cell_list_contents([celda]))]

        if celdas_sucias:
            # Seleccionar la celda más cercana
            celda_mas_cercana = min(celdas_sucias, key=lambda celda: np.linalg.norm(np.array(pos_actual) - np.array(celda)))

            # Limpiar la celda si el robot está sobre una celda sucia
            agentes_en_pos = self.model.grid.get_cell_list_contents([self.pos])
            celda_en_pos = next((agente for agente in agentes_en_pos if isinstance(agente, Celda) and agente.sucia), None)

            if celda_en_pos:
                celda_en_pos.sucia = False

            # Moverse gradualmente hacia la celda más cercana
            posiciones_adyacentes = list(self.model.grid.iter_neighborhood(pos_actual, moore=True))
            posiciones_adyacentes_disponibles = [pos for pos in posiciones_adyacentes if not any(isinstance(agent, (Mueble, RobotLimpieza, Cargador)) for agent in self.model.grid.get_cell_list_contents(pos))]

            if posiciones_adyacentes_disponibles:
                mejor_posicion = min(posiciones_adyacentes_disponibles, key=lambda pos: np.linalg.norm(np.array(celda_mas_cercana) - np.array(pos)))

                # Moverse a la mejor posición adyacente
                self.sig_pos = mejor_posicion
                return True

        return False

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        pos_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        agentes = self.model.grid.get_cell_list_contents(pos_a_limpiar)
        for agente in agentes:
            if isinstance(agente, Celda) and agente.sucia:
                agente.sucia = False
        self.sig_pos = pos_a_limpiar
        
        # Liberar el bloqueo una vez que se ha limpiado la celda
        self.bloqueo = None

    def buscar_celdas_sucia(self, lista_de_vecinos):
        # #Opción 1
        # return [vecino for vecino in lista_de_vecinos
        #                 if isinstance(vecino, Celda) and vecino.sucia]
        # #Opción 2
        celdas_sucias = list()
        for vecino in lista_de_vecinos:
            agentes = self.model.grid.get_cell_list_contents([vecino])
            for agente in agentes:
                if isinstance(agente, Celda) and agente.sucia:
                    celdas_sucias.append(vecino)
        
        return celdas_sucias

    def step(self):
        if(any(isinstance(agent, Cargador) for agent in self.model.grid.get_cell_list_contents(self.pos)) and self.carga < 100):
            self.recargar(25)
            if(self.carga >= 100):
                self.carga = 100
                self.cargasCompletadas += 1


        elif(self.carga <= 30):
            # Encontrar cargador mas cercano
            posiciones_cargadores = [(0,0) , (self.model.grid.width-1,0), (0,self.model.grid.height-1), (self.model.grid.width-1,self.model.grid.height-1)]
            minDistancia = 9999999
            for cargador in posiciones_cargadores:
                distancia = ((cargador[0] - self.pos[0]) ** 2 + (cargador[1] - self.pos[1]) ** 2) ** 0.5
                if(distancia < minDistancia):
                    minDistancia = distancia
                    minCargador = cargador

            #Encontrar siguiente casilla mas cercana al cargador
            pos_vecinos = list(self.model.grid.iter_neighborhood(self.pos, moore=True))
            pos_vecinos_disponibles = [pos for pos in pos_vecinos if not any(isinstance(agent, (Mueble, RobotLimpieza)) for agent in self.model.grid.get_cell_list_contents(pos))]
            minDistancia = 9999999
            for pos in pos_vecinos_disponibles:
                distancia = ((minCargador[0] - pos[0]) ** 2 + (minCargador[1] - pos[1]) ** 2) ** 0.5
                if(distancia < minDistancia):
                    minDistancia = distancia
                    self.sig_pos = pos
        
        else:
            pos_vecinos = list(self.model.grid.iter_neighborhood(self.pos, moore=True))
            pos_vecinos_disponibles = [pos for pos in pos_vecinos if not any(isinstance(agent, (Mueble, RobotLimpieza, Cargador)) for agent in self.model.grid.get_cell_list_contents(pos))]

            celdas_sucias = self.buscar_celdas_sucia(pos_vecinos_disponibles)
            celdas_sucias = [celda for celda in celdas_sucias if celda not in [r.bloqueo for r in self.model.schedule.agents if isinstance(r, RobotLimpieza)]]

            if celdas_sucias:
                distancias = [np.linalg.norm(np.array(self.pos) - np.array(celda)) for celda in celdas_sucias]
                celda_mas_cercana = celdas_sucias[np.argmin(distancias)]

                self.sig_pos = celda_mas_cercana
                self.limpiar_una_celda([celda_mas_cercana])
                self.bloqueo = celda_mas_cercana
            else:
                if not self.buscar_suciedad_extendida():
                    self.sig_pos = self.pos

    def advance(self):
        if self.carga > 0:
            if self.pos != self.sig_pos:
                # Verificar si la próxima posición (sig_pos) está ocupada por otro agente
                agentes_en_sig_pos = self.model.grid.get_cell_list_contents([self.sig_pos])
                robots_en_sig_pos = [agente for agente in agentes_en_sig_pos if isinstance(agente, RobotLimpieza)]

                if not robots_en_sig_pos:
                    # Si no hay otros robots en la próxima posición, moverse
                    self.movimientos += 1
                    self.model.grid.move_agent(self, self.sig_pos)
                    self.carga -= 1
                else:
                    # Si hay otros robots en la próxima posición, no moverse
                    pass
    
class Cargador(Agent):
        def __init__(self, unique_id, model, pos):
            super().__init__(unique_id, model)
            self.pos = pos


class Habitacion(Model):
    def __init__(self, M: int, N: int,
                 num_agentes: int = 5,
                 porc_celdas_sucias: float = 0.6,
                 porc_muebles: float = 0.1,
                 modo_pos_inicial: str = 'Fija',
                 ):

        self.num_agentes = num_agentes
        self.porc_celdas_sucias = porc_celdas_sucias
        self.porc_muebles = porc_muebles

        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter()]

        # Posicionamiento de centros de carga
        num_cargadores = int((M * N) / (M * N / 4))
        posiciones_cargadores = [(0,0) , (M-1,0), (0,N-1), (M-1,N-1)]

        for id, pos in enumerate(posiciones_cargadores):
            cargador = Cargador(int(f"{num_agentes}0{id}") + 1, self, pos)
            self.grid.place_agent(cargador, pos)
            posiciones_disponibles.remove(pos)
            
        # Posicionamiento de muebles
        num_muebles = int(M * N * porc_muebles)
        posiciones_muebles = self.random.sample(posiciones_disponibles, k=num_muebles)

        for id, pos in enumerate(posiciones_muebles):
            mueble = Mueble(int(f"{num_agentes}1{id}") + 1, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)

        # Posicionamiento de celdas sucias
        self.num_celdas_sucias = int(M * N * porc_celdas_sucias)
        posiciones_celdas_sucias = self.random.sample(
            posiciones_disponibles, k=self.num_celdas_sucias)

        for id, pos in enumerate(posiciones_disponibles):
            suciedad = pos in posiciones_celdas_sucias
            celda = Celda(int(f"{num_agentes}{id}") + 1, self, suciedad)
            self.grid.place_agent(celda, pos)

        # Posicionamiento de agentes robot
        if modo_pos_inicial == 'Aleatoria':
            pos_inicial_robots = self.random.sample(posiciones_disponibles, k=num_agentes)
        else:  # 'Fija'
            pos_inicial_robots = [(1, 1)] * num_agentes

        for id in range(num_agentes):
            robot = RobotLimpieza(id, self)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)

        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid, "Cargas": get_todas_cargas, 
                             "Movimientos": get_todos_movimientos, "CeldasSucias": get_sucias},
        )

    def step(self):
        self.datacollector.collect(self)

        self.schedule.step()

    def todoLimpio(self):
        for (content, x, y) in self.grid.coord_iter():
            for obj in content:
                if isinstance(obj, Celda) and obj.sucia:
                    return False
        return True


def get_grid(model: Model) -> np.ndarray:
    """
    Método para la obtención de la grid y representarla en un notebook
    :param model: Modelo (entorno)
    :return: grid
    """
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        x, y = pos
        for obj in cell_content:
            if isinstance(obj, RobotLimpieza):
                grid[x][y] = 2
            elif isinstance(obj, Celda):
                grid[x][y] = int(obj.sucia)
    return grid


def get_cargas(model: Model):
    return [(agent.unique_id, agent.carga) for agent in model.schedule.agents]


def get_sucias(model: Model) -> int:
    """
    Método para determinar el número total de celdas sucias
    :param model: Modelo Mesa
    :return: número de celdas sucias
    """
    sum_sucias = 0
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        for obj in cell_content:
            if isinstance(obj, Celda) and obj.sucia:
                sum_sucias += 1
    return sum_sucias / model.num_celdas_sucias

def get_todas_cargas(model: Model) -> dict:
    return sum([get_cargasCompletadas(agent) for agent in model.schedule.agents])

def get_todos_movimientos(model: Model) -> dict:
    return sum([get_movimientos(agent) for agent in model.schedule.agents])

def get_cargasCompletadas(agent: Agent) -> int:
    if isinstance(agent, RobotLimpieza):
        return agent.cargasCompletadas
    return 0;

def get_movimientos(agent: Agent) -> int:
    if isinstance(agent, RobotLimpieza):
        return agent.movimientos
    else:
        return 0