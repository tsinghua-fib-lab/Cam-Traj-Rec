"""
矩阵分解补全道路速度估计
"""
import pickle
import json
import tqdm
import numpy as np
import requests
import random
from tqdm import tqdm
from math import sqrt
import torch
from collections import defaultdict
from eviltransform import wgs2gcj


G = pickle.load(open("../dataset/road_graph.pkl", "rb"))
edges = list(G.edges(data=True))
N_edge = len(edges)
N_tm = 24
ranked_levels = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "unclassified",
    "living_street",
    "residential",
]
N_feature = len(ranked_levels) + 1
N_latent = 50
user_key = "2d2de838f89274477b723b6eaddd079a"
search_radius = 200
search_portion = [0, 0.25, 0.5, 0.75, 1]
search_weight = [0.26, 0.16, 0.16, 0.16, 0.26]
poi_effect = -0.01

road_speed_simple = pickle.load(open("data/road_speed_simple_full.pkl", "rb"))

level_to_speed = defaultdict(list)
roadid_to_info = {}
for u, v, k in G.edges:
    edge_info = G.edges[u, v, k]
    roadid_to_info[edge_info["id"]] = edge_info
for edge_id, info in roadid_to_info.items():
    level = info["highway"]
    if isinstance(level, str):
        level_to_speed[level].append(road_speed_simple[edge_id])
    else:
        for t in level:
            level_to_speed[t].append(road_speed_simple[edge_id])
level_to_speed = {level: np.mean(speeds) for level, speeds in level_to_speed.items()}
level_effect = []
for level in ranked_levels:
    level_effect.append(level_to_speed[level])


def train_data_matrix():
    M = np.zeros((N_edge, N_tm), dtype=int)
    X_truth = np.zeros((N_edge, N_tm), dtype=float)
    speed_dicts = pickle.load(open("data/road_speed_slice.pkl", "rb"))
    for j in range(N_tm):
        speed_dict = speed_dicts[j]
        print(j, len(speed_dict))
        for i, edge in enumerate(edges):
            id = edge[2]["id"]
            if id in speed_dict:
                X_truth[i][j] = speed_dict[id]
                M[i][j] = 1

    return X_truth, M


def road_feature_matrix():
    def level_rank(level, ranked_levels):
        if isinstance(level, str):
            rank = [ranked_levels.index(level)]
        else:
            rank = [ranked_levels.index(t) for t in level]
        return rank

    F = np.zeros((N_edge, N_feature), dtype=float)
    for i, edge in tqdm(enumerate(edges)):
        info = edge[2]
        rank = level_rank(info["highway"], ranked_levels)
        for r in rank:
            F[i][1 + r] = 1 / len(rank)

        geom = info["geometry"]
        poi_num = 0
        for dist, weight in zip(search_portion, search_weight):
            point = geom.interpolate(dist, normalized=True)
            lon, lat = list(point.coords)[0]
            lat, lon = wgs2gcj(lat, lon)
            lon, lat = round(lon, 6), round(lat, 6)
            url = "https://restapi.amap.com/v3/place/around?key={}&location={},{}&radius={}&output=json&extensions=base&offset=1".format(
                user_key, lon, lat, search_radius
            )
            r = json.loads(requests.get(url).text)
            poi_num += int(r["count"]) * weight
        F[i][0] = poi_num

    return F


def matrix_factorization(X_truth, M, F, device=torch.device("cuda")):
    random.seed(233)
    np.random.seed(233)
    torch.manual_seed(233)
    torch.cuda.manual_seed(233)

    train_portion = 1.0
    t = np.where(M == 1)
    t = list(zip(*t))
    index_train = random.sample(t, round(train_portion * len(t)))
    M_train = np.zeros((N_edge, N_tm), dtype=int)
    M_valid = np.zeros((N_edge, N_tm), dtype=int)
    for i, j in index_train:
        M_train[i][j] = 1
    a = M == 1
    b = M_train == 0
    c = a * b
    M_valid[c] = 1
    assert (M_train + M_valid == M).all()

    X_truth = torch.tensor(X_truth, dtype=torch.float64, device=device)
    M_train = torch.tensor(M_train, dtype=torch.bool, device=device)
    M_valid = torch.tensor(M_valid, dtype=torch.bool, device=device)
    F = torch.tensor(F, dtype=torch.float64, device=device)
    W = torch.tensor(
        [poi_effect] + level_effect,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    U = torch.rand((N_edge, N_latent), device=device, requires_grad=True)
    V = torch.rand((N_tm, N_latent), device=device, requires_grad=True)
    print("F", F.shape, "W", W.shape, "U", U.shape, "V", V.shape)

    N_iter = 9000
    t2_last = 999
    opt = torch.optim.Adam([W, U, V], weight_decay=1e-4)
    with tqdm(range(N_iter)) as tq:
        for i in tq:
            X_rebuild = (F @ W).view(-1, 1) + U @ V.T
            loss = torch.mean(torch.square((X_rebuild - X_truth)[M_train]))
            opt.zero_grad()
            loss.backward()
            opt.step()

            t = loss.cpu().item() ** 0.5
            t2 = (
                torch.mean(torch.square((X_rebuild - X_truth)[M_valid])).cpu().item()
                ** 0.5
            )
            tq.set_description(f"loss: {t:.6f}, valid loss: {t2:.6f}")

    X_rebuild = X_rebuild.cpu().detach().numpy()

    return X_rebuild


if __name__ == "__main__":
    X_truth, M = train_data_matrix()
    F = road_feature_matrix()

    X_rebuild = matrix_factorization(X_truth, M, F)

    X_delta = M * (X_rebuild - X_truth)
    print("MAE:", np.sum(np.abs(X_delta)) / np.sum(M))
    print("RMSE:", sqrt(np.sum(X_delta * X_delta) / np.sum(M)))

    X_simple = np.zeros((N_edge, N_tm), dtype=float)
    speed_dicts = pickle.load(open("data/road_speed_simple_slice.pkl", "rb"))
    for j in range(N_tm):
        speed_dict = speed_dicts[j]
        for i, edge in enumerate(edges):
            X_simple[i][j] = speed_dict[edge[2]["id"]]
    X_delta = (1 - M) * (X_rebuild - X_simple)
    print("MAE:", np.sum(np.abs(X_delta)) / np.sum(1 - M))
    print("RMSE:", sqrt(np.sum(np.square(X_delta)) / np.sum(1 - M)))

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    X_result = np.zeros((N_edge, N_tm), dtype=float)
    for i in range(N_edge):
        for j in range(N_tm):
            if M[i][j] == 1:
                X_result[i][j] = X_truth[i][j]
            else:
                s1 = X_rebuild[i][j]
                s2 = X_simple[i][j]
                s3 = np.sum(F[i][1:] * level_effect)

                if 0.75 * s3 < s2 < 1.25 * s3:
                    s_ref = s2
                elif 0.5 * s3 < s2 < 1.5 * s3:
                    s_ref = (s2 + s3) / 2
                else:
                    s_ref = s3

                if 0.75 * s_ref < s1 < 1.25 * s_ref:
                    s_result = s1
                    cnt1 += 1
                elif 0.5 * s_ref < s1 < 1.5 * s_ref:
                    s_result = (s1 + s_ref) / 2
                    cnt2 += 1
                else:
                    s_result = s_ref
                    cnt3 += 1
                X_result[i][j] = s_result

    print("adopt num:", cnt1)
    print("weak-adopt num:", cnt2)
    print("reject num:", cnt3)

    np.save(file="data/road_speed_completed_slice.npy", arr=X_result)
    speed_result_for_each_slice = [
        {edge[2]["id"]: 0 for edge in edges} for i in range(N_tm)
    ]
    for t in range(N_tm):
        speed_dict = speed_result_for_each_slice[t]
        for edge, speed in zip(speed_dict.keys(), X_result[:, t]):
            speed_dict[edge] = speed
    pickle.dump(
        speed_result_for_each_slice,
        open("../dataset/road_speed_completed_slice.pkl", "wb"),
    )
