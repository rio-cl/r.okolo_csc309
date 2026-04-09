from collections import deque
import heapq

# Graph (city network)
graph = {
    'A': {'B': 2, 'C': 5, 'D': 1},
    'B': {'A': 2, 'D': 2, 'E': 3},
    'C': {'A': 5, 'D': 2, 'F': 3},
    'D': {'A': 1, 'B': 2, 'C': 2, 'E': 1, 'F': 4},
    'E': {'B': 3, 'D': 1, 'G': 2},
    'F': {'C': 3, 'D': 4, 'G': 1},
    'G': {'E': 2, 'F': 1, 'H': 3},
    'H': {'G': 3}
}

# Heuristic values for A*
heuristic = {
    'A': 7,
    'B': 6,
    'C': 6,
    'D': 4,
    'E': 2,
    'F': 2,
    'G': 1,
    'H': 0
}

start = 'A'
goal = 'H'


# 🔷 DFS
def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    nodes_expanded = 0

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes_expanded += 1

        if node == goal:
            return path, len(path) - 1, nodes_expanded

        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, 0, nodes_expanded

# 🔷 BFS
def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_expanded = 0

    while queue:
        node, path = queue.popleft()
        nodes_expanded += 1

        if node == goal:
            return path, len(path) - 1, nodes_expanded

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, 0, nodes_expanded

# 🔷 UCS
def ucs(graph, start, goal):
    pq = [(0, start, [start])]
    best_cost = {start: 0}
    nodes_expanded = 0

    while pq:
        cost, node, path = heapq.heappop(pq)
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded

        for neighbor, weight in graph[node].items():
            new_cost = cost + weight

            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

    return None, 0, nodes_expanded

# 🔷 A* Search
def astar(graph, heuristic, start, goal):
    pq = [(heuristic[start], 0, start, [start])]
    best_cost = {start: 0}
    nodes_expanded = 0

    while pq:
        f, g, node, path = heapq.heappop(pq)
        nodes_expanded += 1

        if node == goal:
            return path, g, nodes_expanded

        for neighbor, weight in graph[node].items():
            new_g = g + weight

            if neighbor not in best_cost or new_g < best_cost[neighbor]:
                best_cost[neighbor] = new_g
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(pq, (new_f, new_g, neighbor, path + [neighbor]))

    return None, 0, nodes_expanded


# 🔷 Run all algorithms
dfs_path, dfs_steps, dfs_nodes = dfs(graph, start, goal)
bfs_path, bfs_steps, bfs_nodes = bfs(graph, start, goal)
ucs_path, ucs_cost, ucs_nodes = ucs(graph, start, goal)
astar_path, astar_cost, astar_nodes = astar(graph, heuristic, start, goal)


# 🔷 Print Results
print("\nDFS:")
print("Path:", dfs_path)
print("Steps:", dfs_steps)
print("Nodes Expanded:", dfs_nodes)
print()

print("BFS:")
print("Path:", bfs_path)
print("Steps:", bfs_steps)
print("Nodes Expanded:", bfs_nodes)
print()

print("UCS:")
print("Path:", ucs_path)
print("Cost:", ucs_cost)
print("Nodes Expanded:", ucs_nodes)
print()

print("A*:")
print("Path:", astar_path)
print("Cost:", astar_cost)
print("Nodes Expanded:", astar_nodes)