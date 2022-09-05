"""
implementation of the iterative framework:
do the clustering and the feedback
"""
import os
import pickle
import time
import random
import numpy as np
import json
from tqdm import tqdm
from math import ceil, sqrt
from multiprocessing import Pool
from cluster_algorithm import SigCluster, FlatSearcher
from eval import evaluate
from itertools import combinations
from copy import deepcopy
from collections import defaultdict
from routing import MAP_routing, MAP_routing_return_route, my_k_shortest_paths


random.seed(233)

cameras = pickle.load(open("../dataset/camera_info.pkl", "rb"))
cameras_dict = {x["id"]: x for x in cameras}

G = pickle.load(open("../dataset/road_graph.pkl", "rb"))

records = pickle.load(open("../dataset/records_100w_pca_64.pkl", "rb"))

f_car = [x["car_feature"] for x in records]
f_plate = [x["plate_feature"] for x in records]
f_emb = deepcopy(f_car)

vid_to_rids = defaultdict(list)
for i, r in enumerate(records):
    t = r["vehicle_id"]
    if t is not None:
        vid_to_rids[t].append(i)

DO_RECALL_ATTEMPT = True
DO_MERGE_ATTEMPT = False
DO_NOISE_SINGLE_CLUSTER = False

MERCLUSTER_SIM_GATE = 0.8
MISS_SHOT_P = 0.6
ADJ_RANGE = 180
MERGE_ATTEMPT_ADJ_RANGE = ADJ_RANGE / 2
TM_GAP_GATE = 720
MERGE_CLUSTER_ADJ_RANGE = 300

workers = 25
ORDINARY_NOISE = 1
STRONG_NOISE = 2
ONE_NOISE = 3
TWO_NOISE = 4
LONG_NOISE = 5
BLACK_LIST_NOISE = 6
OUT_OF_SUBSET_NOISE = 7
SINGLE_CLUSTER = 8


def subsets(arr, k=0, max_return=1000):
    cnt = 0
    if cnt >= max_return:
        return
    for i in range(len(arr), max(0, k - 1), -1):
        for j in combinations(arr, i):
            yield j
            cnt += 1
            if cnt >= max_return:
                return


def merge_tm_adj_points(points, adj_range=ADJ_RANGE):
    node_to_tms = defaultdict(list)
    if isinstance(points[0][-1], list):
        for node, tm, i in points:
            node_to_tms[node].append((tm, i))
    else:
        for node, tm, i in points:
            node_to_tms[node].append((tm, [i]))
    merge_points = []
    for node, tms in node_to_tms.items():
        if len(tms) == 1:
            merge_points.append((node, tms[0][0], tms[0][1]))
        else:
            tms.sort(key=lambda x: x[0])
            min_tm = tms[0][0]
            one_cluster = [tms[0]]
            for tm, i in tms[1:]:
                if tm - min_tm <= adj_range:
                    one_cluster.append((tm, i))
                else:
                    a, b = list(zip(*one_cluster))
                    merge_points.append((node, np.mean(a), sum(b, [])))
                    one_cluster = [(tm, i)]
                    min_tm = tm
            a, b = list(zip(*one_cluster))
            merge_points.append((node, np.mean(a), sum(b, [])))
    return merge_points


def cut_distant_points(points, tm_gap_gate=TM_GAP_GATE):
    cut_points = []
    one_cut = [points[0]]
    tm_last = points[0][1]
    for point in points[1:]:
        tm = point[1]
        if tm - tm_last > tm_gap_gate:
            cut_points.append(one_cut)
            one_cut = [point]
        else:
            one_cut.append(point)
        tm_last = tm
    cut_points.append(one_cut)
    return cut_points


def detect_many_noise(cuts):
    noises = []
    recall_attempts = []
    long_cuts = []
    for i, one_cut in enumerate(cuts):
        if len(one_cut) == 1:
            t = sum([len(cut) for cut in cuts])
            if t == 1 and DO_NOISE_SINGLE_CLUSTER:
                noises.append((one_cut[0][-1], SINGLE_CLUSTER))
            elif t > 6:
                c, ct = one_cut[0][:2]
                flag = True
                if i > 0:
                    u, ut = cuts[i - 1][-1][:2]
                    if MAP_routing(u, c, ut, ct) > 0.1:
                        flag = False
                if flag and i < len(cuts) - 1:
                    v, vt = cuts[i + 1][0][:2]
                    if MAP_routing(c, v, ct, vt) > 0.1:
                        flag = False
                if flag:
                    noises.append((one_cut[0][-1], ONE_NOISE))
        elif len(one_cut) == 2:
            (u, ut, _), (v, vt, _) = one_cut
            p = MAP_routing(u, v, ut, vt)
            if p < 0.05:
                noises += [(x[-1], TWO_NOISE) for x in one_cut]
            elif DO_RECALL_ATTEMPT and p > 0.4:
                inter_nodes, _ = MAP_routing_return_route(u, v, ut, vt)
                inter_camera_nodes = [
                    node for node in inter_nodes if "camera" in G.nodes[node]
                ]
                if inter_camera_nodes:
                    recall_attempts.append(([u, ut], [v, vt], inter_camera_nodes))
        else:
            long_cuts.append(one_cut)

    for one_cut in long_cuts:
        len_cut = len(one_cut)
        p_dict = {}
        sub_ps_raw = []
        sub_ps = []
        sub_idxs = []
        for sub_idx in subsets(list(range(len_cut)), k=ceil(len_cut / 2)):
            ps = []
            for i, j in zip(sub_idx, sub_idx[1:]):
                p = p_dict.get((i, j), None)
                if p is None:
                    u, ut = one_cut[i][:2]
                    v, vt = one_cut[j][:2]
                    p = MAP_routing(u, v, ut, vt)
                    p_dict[(i, j)] = p
                ps.append(p)
            p = np.exp(np.mean(np.log(ps)))
            sub_ps_raw.append(p)
            sub_ps.append(np.exp(np.sum(np.log(ps)) / (len(ps) + 2)))
            sub_idxs.append(sub_idx)

        max_sub_p = max(sub_ps)
        if max_sub_p < 0.05:
            noises += [(x[-1], LONG_NOISE) for x in one_cut]
        elif max_sub_p > 0.45:
            black_list = set()
            white_list = set()
            for i in range(len_cut):
                p = max(p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx)
                if p < 0.01:
                    black_list.add(i)
                    noises.append((one_cut[i][-1], BLACK_LIST_NOISE))
                else:
                    ps = [
                        p
                        for p, idx in zip(sub_ps_raw, sub_idxs)
                        if i in idx and p >= 0.01
                    ]
                    if (
                        len(ps) >= min(len_cut / 3, 2)
                        and np.mean(ps) > 0.1
                        and max(ps) > 0.3
                    ):
                        white_list.add(i)

            opt_sub_p, opt_sub_idx = max(
                (x for x in zip(sub_ps, sub_idxs) if x[0] > 0.8 * max_sub_p),
                key=lambda x: (len(x[1]), x[0]),
            )

            for i in set(range(len_cut)) - set(opt_sub_idx) - black_list - white_list:
                noises.append((one_cut[i][-1], OUT_OF_SUBSET_NOISE))
            if DO_RECALL_ATTEMPT and opt_sub_p > 0.3:
                for i, j in zip(opt_sub_idx, opt_sub_idx[1:]):
                    u, ut, _ = one_cut[i]
                    v, vt, _ = one_cut[j]
                    inter_nodes, _ = MAP_routing_return_route(u, v, ut, vt)
                    inter_camera_nodes = [
                        node for node in inter_nodes if "camera" in G.nodes[node]
                    ]
                    if inter_camera_nodes:
                        recall_attempts.append(((u, ut), (v, vt), inter_camera_nodes))
    if DO_MERGE_ATTEMPT:
        noise_idxss = {tuple(x[0]) for x in noises}
        non_noise_points = [
            (node, tm)
            for one_cut in cuts
            for node, tm, idxs in one_cut
            if tuple(idxs) not in noise_idxss
        ]
        recall_attempts += non_noise_points
    return noises, recall_attempts


def noise_detect_unit(rids):
    points = [
        (cameras_dict[records[i]["camera_id"]]["node_id"], records[i]["time"], i)
        for i in rids
    ]
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points = merge_tm_adj_points(points, adj_range=ADJ_RANGE)
    points.sort(key=lambda x: x[1])
    cuts = cut_distant_points(points, tm_gap_gate=TM_GAP_GATE)
    return detect_many_noise(cuts)


def recall_unit(args):
    def calculate_ave_f(idxs):
        car1 = [f_car[i] for i in idxs]
        plate1 = [f_plate[i] for i in idxs]
        car1 = np.mean(np.asarray(car1), axis=0)
        car1 /= np.linalg.norm(car1) + 1e-12
        plate1 = [x for x in plate1 if x is not None]
        if plate1:
            plate1 = np.mean(np.asarray(plate1), axis=0)
            plate1 /= np.linalg.norm(plate1) + 1e-12
        else:
            plate1 = None
        return car1, plate1

    def sim_filter(car1, plate1, candidates, sim_gate=0.7):
        candidates_filter = []
        for noise in candidates:
            idxs2 = noise[1]
            car2 = [f_car[i] for i in idxs2]
            plate2 = [f_plate[i] for i in idxs2]
            car2 = np.mean(np.asarray(car2), axis=0)
            car2 /= np.linalg.norm(car2) + 1e-12
            plate2 = [x for x in plate2 if x is not None]
            if plate2:
                plate2 = np.mean(np.asarray(plate2), axis=0)
                plate2 /= np.linalg.norm(plate2) + 1e-12
            else:
                plate2 = None
            sim_car = car1 @ car2
            if plate1 is not None and plate2 is not None:
                sim_plate = plate1 @ plate2
                sim = 0.2 * sim_car + 0.8 * sim_plate
            else:
                sim = sim_car
            if sim > sim_gate:
                candidates_filter.append(noise)
        return candidates_filter

    recall_attempts, cr_idxs, node_to_noises = args
    car1, plate1 = None, None
    accept_recalls = []
    for tmp in recall_attempts:
        if len(tmp) == 3:
            (u, ut), (v, vt), inter_camera_nodes = tmp
            p_base = None
            for node in inter_camera_nodes:
                candidates = [
                    noise for noise in node_to_noises[node] if ut < noise[0] < vt
                ]
                if not candidates:
                    continue
                if car1 is None:
                    car1, plate1 = calculate_ave_f(cr_idxs)
                candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=0.7)
                if candidates_filter:
                    if p_base is None:
                        p_base = MAP_routing(u, v, ut, vt)
                    for tm, idxs in candidates_filter:
                        p_new = sqrt(
                            MAP_routing(u, node, ut, tm) * MAP_routing(node, v, tm, vt)
                        )
                        t = p_new * (1 - MISS_SHOT_P) - p_base * MISS_SHOT_P
                        if t > 0:
                            accept_recalls.append((idxs, t))
        else:
            node, tm = tmp
            candidates = [
                noise
                for noise in node_to_noises[node]
                if tm - MERGE_ATTEMPT_ADJ_RANGE
                < noise[0]
                < tm + MERGE_ATTEMPT_ADJ_RANGE
            ]
            if not candidates:
                continue
            if car1 is None:
                car1, plate1 = calculate_ave_f(cr_idxs)
            candidates_filter = sim_filter(car1, plate1, candidates, sim_gate=0.78)
            for tm, idxs in candidates_filter:
                accept_recalls.append((idxs, 0))
    return accept_recalls


def update_f_emb(labels, do_update=True):
    global f_emb

    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    print("detecting noise...")
    start_time = time.time()
    chunksize = min(ceil(len(cid_to_rids) / workers), 200)
    with Pool(processes=workers) as pool:
        results = pool.map(noise_detect_unit, cid_to_rids.values(), chunksize=chunksize)
    cid_to_noises = {}
    cid_to_recall_attempts = {}
    for cid, result in zip(cid_to_rids.keys(), results):
        noises, recall_attempts = result
        if noises:
            cid_to_noises[cid] = noises
        if recall_attempts:
            cid_to_recall_attempts[cid] = recall_attempts
    print("detect noise use time:", time.time() - start_time)

    cid_to_accept_recalls = defaultdict(list)
    if DO_RECALL_ATTEMPT:
        print("recalling...")
        start_time = time.time()
        node_to_noises = defaultdict(list)
        for idxs in (noise[0] for noises in cid_to_noises.values() for noise in noises):
            tms = [records[i]["time"] for i in idxs]
            node = cameras_dict[records[idxs[0]]["camera_id"]]["node_id"]
            node_to_noises[node].append((np.mean(tms), idxs))
        args = [
            (recall_attempts, cid_to_rids[cid], node_to_noises)
            for cid, recall_attempts in cid_to_recall_attempts.items()
        ]
        chunksize = min(ceil(len(args) / workers), 200)
        with Pool(processes=workers) as pool:
            results = pool.map(recall_unit, args, chunksize=chunksize)
        idxs_to_cid_reward = defaultdict(list)
        for cid, accept_recalls in zip(cid_to_recall_attempts.keys(), results):
            if accept_recalls:
                for idxs, reward in accept_recalls:
                    idxs_to_cid_reward[tuple(idxs)].append((cid, reward))
        for idxs, cid_reward in idxs_to_cid_reward.items():
            cid_to_accept_recalls[max(cid_reward, key=lambda x: x[1])[0]] += idxs
        print("recall use time:", time.time() - start_time)

    if not do_update:
        return cid_to_noises, cid_to_accept_recalls

    recalled_noises = []
    to_update = []
    for cid, idxs in cid_to_accept_recalls.items():
        recalled_noises += idxs
        rids = cid_to_rids[cid]
        nids = [y for x in cid_to_noises.get(cid, []) for y in x[0]]
        t = set(rids) - set(nids)
        if len(t) >= len(rids) / 2:
            tmp = [f_emb[i] for i in t]
        else:
            tmp = [f_emb[i] for i in rids]
        tmp = np.mean(np.asarray(tmp), axis=0)
        to_update.append((tmp, idxs))

    for tmp, idxs in to_update:
        for i in idxs:
            f_emb[i] = tmp
    strong_noise_types = {ONE_NOISE, TWO_NOISE, LONG_NOISE, BLACK_LIST_NOISE}
    ordinary_noise_types = {OUT_OF_SUBSET_NOISE}
    for cid, noises in cid_to_noises.items():
        strong_noises = [
            x[0]
            for x in noises
            if x[1] in strong_noise_types and x[0] not in recalled_noises
        ]
        ordinary_noises = [
            x[0]
            for x in noises
            if x[1] in ordinary_noise_types and x[0] not in recalled_noises
        ]
        strong_noises = set(sum(strong_noises, []))
        ordinary_noises = set(sum(ordinary_noises, []))
        noises = strong_noises | ordinary_noises
        pos = cid_to_rids[cid]
        tmp = [f_emb[i] for i in pos if i not in noises]
        if tmp:
            tmp = np.mean(np.asarray(tmp), axis=0)
        else:
            tmp = np.mean(np.asarray([f_emb[i] for i in pos]), axis=0)
        for i in strong_noises:
            f_emb[i] += 0.3 * (f_emb[i] - tmp)
        for i in ordinary_noises:
            f_emb[i] += 0.2 * (f_emb[i] - tmp)
    return cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts


def ave_f_unit(rids):
    if len(rids) == 1:
        return f_car[i], f_plate[i]
    else:
        fs_car = [f_car[i] for i in rids]
        fs_plate = [f_plate[i] for i in rids]
        ave_car = np.mean(np.asarray(fs_car), axis=0)
        fs_plate = [x for x in fs_plate if x is not None]
        if fs_plate:
            ave_plate = np.mean(np.asarray(fs_plate), axis=0)
        else:
            ave_plate = None
        return ave_car, ave_plate


def tms_adj_range(tms, adj_range=MERGE_CLUSTER_ADJ_RANGE):
    tm = tms[0]
    adj_ranges = [[max(tm - adj_range, 0), tm + adj_range]]
    for tm in tms[1:]:
        tm_m = tm - adj_range
        tm_p = tm + adj_range
        if tm_m <= adj_ranges[-1][1]:
            adj_ranges[-1][1] = tm_p
        else:
            adj_ranges.append([tm_m, tm_p])
    return adj_ranges


def merge_cluster_unit(args):
    (c, ncs), cid_to_rids = args
    idxs1 = cid_to_rids[c]
    car1 = [f_car[i] for i in idxs1]
    plate1 = [f_plate[i] for i in idxs1]
    car1 = np.mean(np.asarray(car1), axis=0)
    car1 /= np.linalg.norm(car1) + 1e-12
    plate1 = [x for x in plate1 if x is not None]
    if plate1:
        plate1 = np.mean(np.asarray(plate1), axis=0)
        plate1 /= np.linalg.norm(plate1) + 1e-12
    else:
        plate1 = None
    nidxs_filter = []
    for nc in ncs:
        idxs2 = cid_to_rids[nc]
        for i in idxs2:
            car2 = f_car[i]
            plate2 = f_plate[i]
            car2 /= np.linalg.norm(car2) + 1e-12
            if plate2 is not None:
                plate2 /= np.linalg.norm(plate2) + 1e-12
            sim_car = car1 @ car2
            if plate1 is not None and plate2 is not None:
                sim_plate = plate1 @ plate2
                sim = 0.2 * sim_car + 0.8 * sim_plate
            else:
                sim = sim_car
            if sim > MERCLUSTER_SIM_GATE:
                nidxs_filter.append(i)
    if not nidxs_filter:
        return []

    points = [
        (cameras_dict[records[i]["camera_id"]]["node_id"], records[i]["time"], i)
        for i in cid_to_rids[c]
    ]
    tm_ranges = tms_adj_range(
        sorted([x[1] for x in points]), adj_range=MERGE_CLUSTER_ADJ_RANGE
    )
    points_nc = [
        (cameras_dict[records[i]["camera_id"]]["node_id"], records[i]["time"], i)
        for i in nidxs_filter
    ]
    points_nc_filter = []
    for p in points_nc:
        t = p[1]
        flag = False
        for min_t, max_t in tm_ranges:
            if min_t < t < max_t:
                flag = True
                break
        if flag:
            points_nc_filter.append(p)
    if not points_nc_filter:
        return []
    points_nc = points_nc_filter

    points_all = points + points_nc
    points_all = merge_tm_adj_points(points_all, adj_range=ADJ_RANGE)
    points_all = merge_tm_adj_points(points_all, adj_range=ADJ_RANGE)
    points_all.sort(key=lambda x: x[1])
    cuts = cut_distant_points(points_all, tm_gap_gate=TM_GAP_GATE)

    noises = []
    long_cuts = []
    for i, one_cut in enumerate(cuts):
        if len(one_cut) == 1:
            noises.append(one_cut[0][-1])
        elif len(one_cut) == 2:
            (u, ut, _), (v, vt, _) = one_cut
            p = MAP_routing(u, v, ut, vt)
            if p < 0.3:
                noises += [x[-1] for x in one_cut]
        else:
            long_cuts.append(one_cut)
    for one_cut in long_cuts:
        len_cut = len(one_cut)
        p_dict = {}
        sub_ps_raw = []
        sub_ps = []
        sub_idxs = []
        for sub_idx in subsets(list(range(len_cut)), k=ceil(len_cut * 3 / 5)):
            ps = []
            for i, j in zip(sub_idx, sub_idx[1:]):
                p = p_dict.get((i, j), None)
                if p is None:
                    u, ut = one_cut[i][:2]
                    v, vt = one_cut[j][:2]
                    p = MAP_routing(u, v, ut, vt)
                    p_dict[(i, j)] = p
                ps.append(p)
            p = np.exp(np.mean(np.log(ps)))
            sub_ps_raw.append(p)
            sub_ps.append(np.exp(np.sum(np.log(ps)) / (len(ps) + 2)))
            sub_idxs.append(sub_idx)

        max_sub_p = max(sub_ps)
        if max_sub_p < 0.3:
            noises += [x[-1] for x in one_cut]
        else:
            black_list = set()
            white_list = set()
            for i in range(len_cut):
                p = max(p for p, idx in zip(sub_ps_raw, sub_idxs) if i in idx)
                if p < 0.01:
                    black_list.add(i)
                    noises.append(one_cut[i][-1])
                else:
                    ps = [
                        p
                        for p, idx in zip(sub_ps_raw, sub_idxs)
                        if i in idx and p >= 0.01
                    ]
                    if (
                        len(ps) >= min(len_cut / 3, 3)
                        and np.mean(ps) > 0.2
                        and max(ps) > 0.5
                    ):
                        white_list.add(i)
            opt_sub_p, opt_sub_idx = max(
                (x for x in zip(sub_ps, sub_idxs) if x[0] > 0.8 * max_sub_p),
                key=lambda x: (len(x[1]), x[0]),
            )
            for i in set(range(len_cut)) - set(opt_sub_idx) - black_list - white_list:
                noises.append(one_cut[i][-1])
    noise_idxs = {idx for idxs in noises for idx in idxs}
    orig_idxs = {x[-1] for x in points}
    merge_point_idxs = set()
    for p in points_all:
        idxs = {x for x in p[-1]}
        tmp = idxs - orig_idxs
        if len(tmp) < len(idxs):
            for idx in tmp:
                merge_point_idxs.add(idx)
    accept_idxs = {x[-1] for x in points_nc} - noise_idxs - merge_point_idxs
    return list(accept_idxs)


def merge_clusters(labels, ngpu=1):
    global f_emb
    print("cluster merging...")
    start_time = time.time()

    cid_to_rids = defaultdict(list)
    for i, c in enumerate(labels):
        if c != -1:
            cid_to_rids[c].append(i)

    for nn in range(3):
        if nn == 0:
            cs_big = []
            cs_small = []
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 10 < t <= 20:
                    cs_big.append(c)
                elif t <= 10:
                    cs_small.append(c)
        elif nn == 1:
            cs_big = []
            cs_small = []
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 20 < t <= 30:
                    cs_big.append(c)
                elif t <= 20:
                    cs_small.append(c)
        elif nn == 2:
            cs_big = []
            cs_small = []
            for c, rids in cid_to_rids.items():
                t = len(rids)
                if 30 < t:
                    cs_big.append(c)
                elif t <= 30:
                    cs_small.append(c)

        args = [cid_to_rids[c] for c in cs_big]
        with Pool(processes=workers) as pool:
            results = pool.map(
                ave_f_unit, args, chunksize=min(ceil(len(args) / workers), 100)
            )
        car_query = np.asarray([x[0] for x in results])
        tmp = [(x[1], c) for x, c in zip(results, cs_big) if x[1] is not None]
        plate_query = np.asarray([x[0] for x in tmp])
        plate_query_c = [x[1] for x in tmp]

        args = [cid_to_rids[c] for c in cs_small]
        with Pool(processes=workers) as pool:
            results = pool.map(
                ave_f_unit, args, chunksize=min(ceil(len(args) / workers), 500)
            )
        car_gallery = np.asarray([x[0] for x in results])
        tmp = [(x[1], c) for x, c in zip(results, cs_small) if x[1] is not None]
        plate_gallery = np.asarray([x[0] for x in tmp])
        plate_gallery_c = [x[1] for x in tmp]

        car_topk = 15
        plate_topk = 30
        car_searcher = FlatSearcher(feat_len=64, ngpu=ngpu)
        plate_searcher = FlatSearcher(feat_len=64, ngpu=ngpu)
        car_topk_idxs = car_searcher.search_by_topk(
            query=car_query, gallery=car_gallery, topk=car_topk
        )[1].tolist()
        plate_topk_idxs = plate_searcher.search_by_topk(
            query=plate_query, gallery=plate_gallery, topk=plate_topk
        )[1].tolist()
        c_to_nc = defaultdict(set)
        for c, idxs in zip(cs_big, car_topk_idxs):
            for i in idxs:
                c_to_nc[c].add(cs_small[i])
        for c, idxs in zip(plate_query_c, plate_topk_idxs):
            for i in idxs:
                c_to_nc[c].add(plate_gallery_c[i])

        args = [((c, ncs), cid_to_rids) for c, ncs in c_to_nc.items()]
        with Pool(processes=workers) as pool:
            results = pool.map(
                merge_cluster_unit, args, chunksize=min(ceil(len(args) / workers), 200)
            )
        accept_idx_to_cs = defaultdict(list)
        for c, accept_idxs in zip(c_to_nc.keys(), results):
            for idx in accept_idxs:
                accept_idx_to_cs[idx].append(c)
        c_to_accept_idxs = defaultdict(list)
        for idx, cs in accept_idx_to_cs.items():
            if len(cs) == 1:
                c_to_accept_idxs[cs[0]].append(idx)
            else:
                c_to_accept_idxs[random.sample(cs, 1)[0]].append(idx)

        for c, idxs in c_to_accept_idxs.items():
            for idx in idxs:
                labels[idx] = c
        for c, idxs in c_to_accept_idxs.items():
            tmp = [f_emb[i] for i in cid_to_rids[c]]
            tmp = np.mean(np.asarray(tmp), axis=0)
            tmp2 = [f_emb[i] for i in idxs]
            tmp2 = np.mean(np.asarray(tmp2), axis=0)
            delta = tmp - tmp2
            for idx in idxs:
                f_emb[idx] += delta

    print("merging consume time:", time.time() - start_time)
    return labels


if __name__ == "__main__":
    N_iter = 10
    s = 0.8
    topK = 128
    ngpu = 3
    metrics = []

    cache_path = "data/shortest_path_results_test.pkl"
    if not os.path.exists(cache_path):
        print("pre-computing...")
        camera_nodes = [x["node_id"] for x in cameras]
        camera_nodes = set(camera_nodes)
        shortest_path_results = {}
        for u in tqdm(camera_nodes):
            for v in camera_nodes:
                if u != v:
                    try:
                        paths = [x for x in my_k_shortest_paths(u, v, 10)]
                        shortest_path_results[(u, v)] = paths
                    except:
                        pass
        print(len(shortest_path_results))
        pickle.dump(shortest_path_results, open(cache_path, "wb"))

    for i, operation in zip(range(N_iter), ["merge", "denoise"] * (N_iter // 2)):

        print(f"---------- iter {i} -----------")

        if i > -1:
            print("clustering...")
            start_time = time.time()
            cluster = SigCluster(feature_dims=[64, 64, 64], ngpu=ngpu)
            labels = cluster.fit(
                [[a, b, c] for a, b, c in zip(f_car, f_plate, f_emb)],
                weights=[0.1, 0.8, 0.1],
                similarity_threshold=s,
                topK=topK,
            )
            print("clustering consume time:", time.time() - start_time)
            pickle.dump(labels, open(f"label/labels_iter_{i}.pkl", "wb"))
        else:
            labels = pickle.load(open(f"label/labels_iter_{i}.pkl", "rb"))

        precision, recall, fscore, expansion, vid_to_cid = evaluate(records, labels)
        metrics.append((precision, recall, fscore, expansion))

        if operation == "merge":
            merge_clusters(labels, ngpu=ngpu)
        elif operation == "denoise":
            cid_to_noises, cid_to_accept_recalls, cid_to_recall_attempts = update_f_emb(
                labels, do_update=True
            )

    json.dump(metrics, open(f"metric/metrics.json", "w"))
