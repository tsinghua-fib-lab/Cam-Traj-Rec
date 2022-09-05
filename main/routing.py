"""
the implementation of prior/likelihood/posterior probilities
infer the MAP path
"""
import pickle
import itertools
import osmnx as ox
import numpy as np
from math import exp
from networkx import shortest_simple_paths
from collections import defaultdict


G = pickle.load(open("../dataset/road_graph.pkl", "rb"))
G_di = ox.utils_graph.get_digraph(G, "length")

K = 10
SIGMA_RATIO = 0.8
PRIOR_W = 5
SPEED_MULTIPLE = 1.25

speed_dicts = pickle.load(open("../dataset/road_speed_completed_slice.pkl", "rb"))
for speed_dict in speed_dicts:
    for key, value in speed_dict.items():
        speed_dict[key] = value * SPEED_MULTIPLE

camera_node_to_node_to_A = pickle.load(
    open("../dataset/camera_node_to_node_to_A.pkl", "rb")
)
camera_node_to_node_to_A_p = pickle.load(
    open("../dataset/camera_node_to_node_to_A_p.pkl", "rb")
)
node_to_A = pickle.load(open("../dataset/node_to_A.pkl", "rb"))
node_to_A_p = pickle.load(open("../dataset/node_to_A_p.pkl", "rb"))

edge_info_dict = {}
for u, v, k in G.edges:
    edge_info = G.edges[u, v, k]
    edge_info_dict[edge_info["id"]] = [u, v, k, edge_info]

edge_to_pred_succ_index = defaultdict(dict)
for node, info in G.nodes(data=True):
    for i, edge in enumerate(info["pred"]):
        edge_to_pred_succ_index[edge]["pred"] = i
    for i, edge in enumerate(info["succ"]):
        edge_to_pred_succ_index[edge]["succ"] = i

shortest_path_results = pickle.load(open("data/shortest_path_results.pkl", "rb"))

def gauss(v, mu):
    sigma = mu * SIGMA_RATIO
    return exp(-((v - mu) ** 2) / sigma**2 / 2)


def route_likelihood(route, ttm, slot, route_type="node"):
    speed_dict = speed_dicts[slot]
    total_etm = 0
    if route_type == "node":
        edges = []
        for n1, n2 in zip(route, route[1:]):
            edge = G.edges[n1, n2, 0]
            length = edge["length"]
            edge_id = edge["id"]
            speed = speed_dict[edge_id]
            etm = length / speed
            total_etm += etm
            edges.append(edge_id)
        v = length / (ttm * etm / total_etm)
        return gauss(v, speed), edges
    elif route_type == "edge":
        for edge in route:
            length = edge_info_dict[edge][-1]["length"]
            speed = speed_dict[edge]
            etm = length / speed
            total_etm += etm
        v = length / (ttm * etm / total_etm)
        return gauss(v, speed)


def route_prior(route, return_p_nostart=False):
    u = edge_info_dict[route[0]][0]
    v = edge_info_dict[route[-1]][1]
    A_dict = camera_node_to_node_to_A.get(v, node_to_A)
    A_dict_p = camera_node_to_node_to_A_p.get(v, node_to_A_p)
    if u in A_dict:
        tmp = np.sum(A_dict[u], axis=0)
        tmp += np.ones(tmp.shape) * PRIOR_W * A_dict[u].shape[0]
        tmp /= np.sum(tmp)
        p_start = tmp[edge_to_pred_succ_index[route[0]]["succ"]]
    else:
        p_start = 1 / len(G.nodes[u]["succ"])
    p_nostart = 1.0
    nodes = [edge_info_dict[x][0] for x in route[1:]]
    for rin, rout, node in zip(route, route[1:], nodes):
        A = A_dict_p.get(node, node_to_A_p.get(node, None))
        if A is None:
            p_nostart *= 1 / len(G.nodes[node]["succ"])
        else:
            p_nostart *= A[edge_to_pred_succ_index[rin]["pred"]][
                edge_to_pred_succ_index[rout]["succ"]
            ]
    if return_p_nostart:
        return p_start * p_nostart, p_nostart
    else:
        return p_start * p_nostart


def my_k_shortest_paths(u, v, k):
    paths_gen = shortest_simple_paths(G_di, u, v, "length")
    for path in itertools.islice(paths_gen, 0, k):
        yield path


def read_k_shortest_path(u, v, k):
    t = shortest_path_results.get((u, v), [])
    return t[:k]


def MAP_routing(u, v, ut, vt, k=K):
    ttm = vt - ut
    if ttm == 0:
        ttm += 0.01
    slot = int(ut / 3600)
    assert ttm > 0
    if u != v:
        proposals = read_k_shortest_path(u, v, k)
    else:
        proposals = []
        for inter in G[u].keys():
            tmp = my_k_shortest_paths(inter, v, 5)
            for t in tmp:
                proposals.append([u] + t)
    if not proposals:
        return 1e-12

    posteriors = []
    for nodes in proposals:
        likelihoood, route = route_likelihood(nodes, ttm, slot)
        prior = route_prior(route)
        posteriors.append(likelihoood * prior)
    return max(max(posteriors), 1e-12)


def MAP_routing_return_route(u, v, ut, vt, k=K):
    ttm = vt - ut
    if ttm == 0:
        ttm += 0.01
    slot = int(ut / 3600)
    assert ttm > 0
    if u != v:
        proposals = read_k_shortest_path(u, v, k)
    else:
        proposals = []
        for inter in G[u].keys():
            tmp = my_k_shortest_paths(inter, v, 5)
            for t in tmp:
                proposals.append([u] + t)
    if not proposals:
        return [], 1e-12

    posteriors = []
    routes = []
    for nodes in proposals:
        likelihoood, route = route_likelihood(nodes, ttm, slot)
        prior = route_prior(route)
        posteriors.append(likelihoood * prior)
        routes.append(nodes[1:-1])
    t = posteriors.index(max(posteriors))
    return routes[t], max(posteriors[t], 1e-12)


def MAP_routing_return_edge_route(u, v, ut, vt, k=K):
    ttm = vt - ut
    if ttm == 0:
        ttm += 0.01
    slot = int(ut / 3600)
    assert ttm > 0
    if u != v:
        proposals = read_k_shortest_path(u, v, k)
    else:
        proposals = []
        for inter in G[u].keys():
            tmp = my_k_shortest_paths(inter, v, 5)
            for t in tmp:
                proposals.append([u] + t)
    if not proposals:
        return [], 1e-12

    posteriors = []
    routes = []
    for nodes in proposals:
        likelihoood, route = route_likelihood(nodes, ttm, slot)
        prior = route_prior(route)
        posteriors.append(likelihoood * prior)
        routes.append(route)
    t = posteriors.index(max(posteriors))
    return routes[t], max(posteriors[t], 1e-12)
