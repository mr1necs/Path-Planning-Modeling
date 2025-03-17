import networkx as nx
import matplotlib.pyplot as plt
import heapq
import random
import math


class Path:
    def __init__(self, cities_list, num_edges):
        self.cities = cities_list
        self.num_edges = num_edges
        self.G = nx.Graph()
        self.generate_edges()


    def generate_edges(self):
        edges = set()
        while len(edges) < self.num_edges:
            city_pair = tuple(sorted(random.sample(self.cities, 2)))
            if city_pair not in edges:
                weight = random.randint(500, 2000)
                self.G.add_edge(*city_pair, weight=weight)
                edges.add(city_pair)


    def reconstruct_path(self, parent, start, goal):
        if goal not in parent:
            return []
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path if path and path[0] == start else []


    def dijkstra(self, start, goal):
        queue = [(0, start)]
        g = {city: float('inf') for city in self.cities}
        g[start] = 0
        parent = {start: None}
        visited_order = []

        while queue:
            cost, current = heapq.heappop(queue)
            if cost > g[current]:
                continue
            visited_order.append(current)
            if current == goal:
                break
            for neighbor in self.G.neighbors(current):
                new_cost = cost + self.G[current][neighbor]['weight']
                if new_cost < g[neighbor]:
                    g[neighbor] = new_cost
                    parent[neighbor] = current
                    heapq.heappush(queue, (new_cost, neighbor))
        return self.reconstruct_path(parent, start, goal), visited_order


    def a_star(self, start, goal):
        pos = nx.spring_layout(self.G, seed=42)
        heuristic = lambda n: math.hypot(pos[n][0] - pos[goal][0], pos[n][1] - pos[goal][1])
        open_set = [(heuristic(start), start)]
        g = {city: float('inf') for city in self.cities}
        g[start] = 0
        parent = {start: None}
        visited_order = []
        closed_set = set()

        while open_set:
            f, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            visited_order.append(current)
            if current == goal:
                break
            closed_set.add(current)
            for neighbor in self.G.neighbors(current):
                if neighbor in closed_set:
                    continue
                tentative_g = g[current] + self.G[current][neighbor]['weight']
                if tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    parent[neighbor] = current
                    heapq.heappush(open_set, (tentative_g + heuristic(neighbor), neighbor))
        return self.reconstruct_path(parent, start, goal), visited_order


    def animate(self, visited_order, path, visited_node_color, visited_edge_color):
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(self.G, seed=42)


        def update(frame):
            ax.clear()
            nx.draw(self.G, pos, with_labels=True, node_color="lightgray", edge_color="gray",
                    node_size=2000, font_size=10)
            current_nodes = visited_order[:frame + 1]
            nx.draw_networkx_nodes(self.G, pos, nodelist=current_nodes,
                                   node_color=visited_node_color, node_size=2000)
            for i in range(len(current_nodes) - 1):
                nx.draw_networkx_edges(self.G, pos,
                                       edgelist=[(current_nodes[i], current_nodes[i + 1])],
                                       edge_color=visited_edge_color, width=2)
                if i == len(current_nodes) - 2:
                    print(f"Переход: {current_nodes[i]} -> {current_nodes[i + 1]}")

            if frame == len(visited_order) - 1 and path:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(self.G, pos, edgelist=path_edges,
                                       edge_color="red", width=3)

        for i in range(len(visited_order)):
            update(i)
            plt.pause(0.5)
        plt.show()


    def animate_dijkstra(self, start, goal):
        path, visited_order = self.dijkstra(start, goal)
        self.animate(visited_order, path, visited_node_color="lightblue", visited_edge_color="lightblue")
        return visited_order


    def animate_a_star(self, start, goal):
        path, visited_order = self.a_star(start, goal)
        self.animate(visited_order, path, visited_node_color="lightgreen", visited_edge_color="lightgreen")
        return visited_order


if __name__ == '__main__':
    cities = [
        "Москва", "Лондон", "Берлин", "Бостон", "Мадрид",
        "Даллас", "Чикаго", "Женева", "Глазго", "Сидней",
        "Анкара", "Дублин", "Брюгге", "Мюнхен", "Таллин"
    ]

    graph = Path(cities, num_edges=30)
    start_city, goal_city = random.sample(cities, 2)

    print("\nАнимация алгоритма Дейкстры:")
    visited_dijkstra = graph.animate_dijkstra(start=start_city, goal=goal_city)

    print("\nАнимация алгоритма A*:")
    visited_a_star = graph.animate_a_star(start=start_city, goal=goal_city)

    print("\nПосещенные города (Дейкстра) :", ", ".join(visited_dijkstra))
    print("Посещенные города (A*)       :", ", ".join(visited_a_star))