"""Physical distance paths and geometry-informed (not observed road) networks."""
import heapq
import math
from fractions import Fraction
import networkx as nx


def fixed_paths(graph):
    """Dijkstra with exact decimal edge sums, then unique additive edge-set rank.

    Positive edges make every shortest path simple. Binary edge ranks select a
    unique path among exact distance ties, symmetrically and with subpath consistency.
    Secondary ranks never perturb primary physical distance.
    """
    if not nx.is_connected(graph):
        raise ValueError("network must be connected")
    edges = sorted((min(str(a), str(b)), max(str(a), str(b))) for a, b in graph.edges)
    ranks = {e: 1 << i for i, e in enumerate(edges)}
    paths, distances = {}, {}
    for source in sorted(graph, key=str):
        best = {source: (Fraction(0), 0)}
        chosen = {source: [source]}
        queue = [(Fraction(0), 0, str(source), source)]
        while queue:
            dist, rank, _, u = heapq.heappop(queue)
            if (dist, rank) != best[u]:
                continue
            for v in sorted(graph[u], key=str):
                w = graph[u][v]["weight"]
                if not math.isfinite(w) or w <= 0:
                    raise ValueError("physical edge lengths must be finite and positive")
                key = tuple(sorted((str(u), str(v))))
                value = (dist + Fraction(str(w)), rank + ranks[key])
                if v not in best or value < best[v]:
                    best[v], chosen[v] = value, chosen[u] + [v]
                    heapq.heappush(queue, (*value, str(v), v))
        for v in graph:
            paths[source, v], distances[source, v] = chosen[v], float(best[v][0])
    return paths, distances


def eligible_pickups(net, hub, cross=True, radius=0, rule="network_radius"):
    if radius < 0 or not math.isfinite(radius):
        raise ValueError("pickup radius must be finite and nonnegative")
    if rule not in {"network_radius", "legacy_adjacency"}:
        raise ValueError("unknown pickup rule")
    if not cross:
        return [hub]
    others = (list(net.H.neighbors(hub)) if rule == "legacy_adjacency" else
              [v for v in net.H if v != hub and net.d[hub, v] <= radius])
    return [hub] + sorted(others, key=str)


def geometric_network(coordinates, k=1, stretch=1.2):
    """Local equirectangular mile coordinates, deterministic kNN + MST.

    Input lat/lon are explicit representative-location assumptions, not CFS centroids.
    Stretch is a modeling assumption, not observed highway circuity.
    """
    if stretch < 1 or not math.isfinite(stretch) or k < 1 or len(coordinates) < 2:
        raise ValueError("need >=2 hubs, k>=1 and finite stretch>=1")
    names = sorted(coordinates)
    lat0 = math.radians(sum(coordinates[n][0] for n in names) / len(names))
    lon0 = sum(coordinates[n][1] for n in names) / len(names)
    xy = {n: [3958.7613 * math.radians(coordinates[n][1] - lon0) * math.cos(lat0),
              3958.7613 * (math.radians(coordinates[n][0]) - lat0)] for n in names}
    complete = nx.Graph()
    complete.add_nodes_from(names)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            distance = math.dist(xy[a], xy[b])
            if distance == 0:
                raise ValueError("representative hub locations must be distinct")
            complete.add_edge(a, b, weight=distance)
    graph = nx.minimum_spanning_tree(complete)
    for a in names:
        for b in sorted((b for b in names if b != a), key=lambda b: (complete[a][b]["weight"], b))[:k]:
            graph.add_edge(a, b, weight=complete[a][b]["weight"])
    for a, b in graph.edges:
        graph[a][b]["weight"] = complete[a][b]["weight"] * stretch
    return graph, xy
