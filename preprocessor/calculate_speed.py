"""
calculate road speed based on map-matched historical trajectories
"""
import pickle
from collections import defaultdict
from copy import deepcopy
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm


road_path = "../dataset/road_graph.pkl"
matched_traj_path = "data/matched_traj.pkl"

MIN_SPEED = 1
MIN_SPEED2 = 3
MAX_SPEED = 33
MAX_SPEED2 = 28
DIS_GATE = 8
DIS_GATE2 = 8
FAKE_SPEED = 10
ADJ_GATE = 1
METHOD_WEIGHT = [0.45, 0.35, 0.2]


def preprocess(traj_match, roadid_to_info):
    result = []
    for traj in tqdm(traj_match):
        edges = traj["path"]
        for edge in edges:
            edge_id = edge[0]
            info = roadid_to_info[edge_id]
            geom = info["geometry"]
            length = info["length"]

            points = edge[1]
            points_new = []
            for point in points:
                lon, lat, tm = point["point"]
                dist = geom.project(Point(lon, lat), normalized=True)
                dist *= length
                points_new.append([dist, tm])
            edge[1] = points_new
        result.append(edges)
    return result


def method1(road):
    record = road[1]
    if len(record) < 2:
        print("point number < 2!")
        return -1

    ss = []
    total_t = 0
    for (x1, t1), (x2, t2) in zip(record, record[1:]):
        dx = x2 - x1
        dt = t2 - t1
        if dt > 0 and dx >= 0:
            s = dx / dt
            if MIN_SPEED < s < MAX_SPEED:
                ss.append((s, dt))
                total_t += dt
    if ss:
        s = sum([a * b for a, b in ss]) / total_t
    else:
        return -1

    if s < MIN_SPEED2 and total_t < 20:
        return -1
    if s > MAX_SPEED2 and total_t < 20:
        return -1

    t = total_t / len(ss)
    if t < 11:
        weight = 1
    elif t < 21:
        weight = 0.8
    elif t < 31:
        weight = 0.6
    else:
        weight = 0.4

    return [s, weight]


def method2(road, road_prev, road_next, roadid_to_info):
    if len(road[1]) == 0 or len(road_prev[1]) == 0 or len(road_next[1]) == 0:
        print("point number == 0!")
        return -1

    length = roadid_to_info[road[0]]["length"]
    length_prev = roadid_to_info[road_prev[0]]["length"]

    x_first, t_first = road[1][0]
    x_last, t_last = road[1][-1]
    x_prev, t_prev = road_prev[1][-1]
    x_next, t_next = road_next[1][0]

    dis1 = max(length_prev - x_prev, 0)
    dis2 = x_first
    dis3 = max(length - x_last, 0)
    dis4 = x_next

    try:
        t_in = (dis1 * t_first + dis2 * t_prev) / (dis1 + dis2)
    except:
        t_in = (t_first + t_prev) / 2
    try:
        t_out = (dis3 * t_next + dis4 * t_last) / (dis3 + dis4)
    except:
        t_out = (t_next + t_last) / 2
    dt = t_out - t_in
    if dt <= 0:
        return -1
    s = length / dt

    span = (t_first - t_prev) + (t_next - t_last)

    if not MIN_SPEED < s < MAX_SPEED:
        return -1
    if s < MIN_SPEED2 and span > 60:
        return -1
    if s > MAX_SPEED2 and span > 60:
        return -1

    if span < 10:
        weight = 1
    elif span < 30:
        weight = 0.8
    elif span < 50:
        weight = 0.6
    else:
        weight = 0.4

    return [s, weight]


def method3(i, roadid_to_info, roads):
    len_roads = len(roads)
    road = roads[i]
    results = []
    length = roadid_to_info[road[0]]["length"]

    if road[1]:
        x_orig, t_orig = road[1][0]
        x_dest, t_dest = road[1][-1]
        dis_orig = x_orig
        dis_dest = max(length - x_dest, 0)

        if dis_dest < DIS_GATE or dis_dest < length / DIS_GATE2:
            if not i == 0:
                road_prev = roads[i - 1]
                if not road_prev[1] == []:
                    length_prev = roadid_to_info[road_prev[0]]["length"]
                    x_dest_prev, t_dest_prev = road_prev[1][-1]
                    dis_dest_prev = max(length_prev - x_dest_prev, 0)
                    if (
                        dis_dest_prev < DIS_GATE
                        or dis_dest_prev < length_prev / DIS_GATE2
                    ):
                        est_time = (t_dest - t_dest_prev) + (
                            dis_dest - dis_dest_prev
                        ) / FAKE_SPEED
                        s = length / est_time
                        if not (s < MIN_SPEED or s > MAX_SPEED):
                            results.append(s)

        if dis_orig < DIS_GATE or dis_orig < length / DIS_GATE2:
            if not i == len_roads - 1:
                road_next = roads[i + 1]
                if not road_next[1] == []:
                    length_next = roadid_to_info[road_next[0]]["length"]
                    x_orig_next, t_orig_next = road_next[1][0]
                    dis_orig_next = x_orig_next
                    if (
                        dis_orig_next < DIS_GATE
                        or dis_orig_next < length_next / DIS_GATE2
                    ):
                        est_time = (t_orig_next - t_orig) + (
                            dis_orig - dis_orig_next
                        ) / FAKE_SPEED
                        s = length / est_time
                        if not (s < MIN_SPEED or s > MAX_SPEED):
                            results.append(s)

    if not (i == 0 or i == len_roads - 1):
        road_prev = roads[i - 1]
        road_next = roads[i + 1]
        if not (road_next[1] == [] or road_prev[1] == []):
            length_prev = roadid_to_info[road_prev[0]]["length"]
            length_next = roadid_to_info[road_next[0]]["length"]
            x_dest_prev, t_dest_prev = road_prev[1][-1]
            x_orig_next, t_orig_next = road_next[1][0]
            dis_dest_prev = max(length_prev - x_dest_prev, 0)
            dis_orig_next = x_orig_next
            if dis_dest_prev < DIS_GATE or dis_dest_prev < length_prev / DIS_GATE2:
                if dis_orig_next < DIS_GATE or dis_orig_next < length_next / DIS_GATE2:
                    est_time = (t_orig_next - t_dest_prev) - (
                        dis_dest_prev + dis_orig_next
                    ) / FAKE_SPEED
                    s = length / est_time
                    if not (s < MIN_SPEED or s > MAX_SPEED):
                        results.append(s)

    if results:
        return np.mean(results)
    else:
        return -1


def remove_extreme(speed):
    if len(speed) < 6:
        return speed
    mean = np.mean(np.array([spd[0] for spd in speed]))
    std = np.std(np.array([spd[0] for spd in speed]))
    multi = 1
    len_orig = len(speed)
    while True:
        if multi > 4:
            return speed
        tmp = [
            spd
            for spd in speed
            if spd[0] > mean - multi * std and spd[0] < mean + multi * std
        ]
        if len(tmp) > 0.7 * len_orig:
            return tmp
        else:
            multi = multi + 0.1


def speed_estimate(traj_match, roadid_to_info):
    speeds_by_road = {}
    method1_cnt = 0
    method2_cnt = 0
    method3_cnt = 0
    complement_cnt = 0
    for roads in tqdm(traj_match, desc="speed_estimate"):
        len_roads = len(roads)
        speed_roads = [
            {"method1": None, "method2": None, "method3": None, "complement": None}
            for i in range(len_roads)
        ]

        for i in range(len_roads):
            road = roads[i]

            if len(road[1]) > 1:
                s1 = method1(road)
                if not s1 == -1:
                    speed_roads[i]["method1"] = s1
                    method1_cnt += 1

            if not (i == 0 or i == len_roads - 1):
                road_prev = roads[i - 1]
                road_next = roads[i + 1]
                if len(road[1]) > 0 and len(road_prev[1]) > 0 and len(road_next[1]) > 0:
                    s2 = method2(road, road_prev, road_next, roadid_to_info)
                    if not s2 == -1:
                        speed_roads[i]["method2"] = s2
                        method2_cnt += 1

            s3 = method3(i, roadid_to_info, roads)
            if not s3 == -1:
                speed_roads[i]["method3"] = [s3, 1]
                method3_cnt += 1

        for i in range(len_roads):
            tmp = speed_roads[i]
            if (
                tmp["method1"] is None
                and tmp["method2"] is None
                and tmp["method3"] is None
            ):
                speeds_adj = []
                for j in range(i - ADJ_GATE, i + ADJ_GATE + 1):
                    if 0 <= j < len_roads and j != i:
                        for key, value in speed_roads[j].items():
                            if key != "complement":
                                if value is not None:
                                    speeds_adj.append(value)
                if speeds_adj:
                    total_s = 0
                    total_w = 0
                    for s, w in speeds_adj:
                        total_s += s * w
                        total_w += w
                    speed_roads[i]["complement"] = [total_s / total_w, 1]
                    complement_cnt = complement_cnt + 1

        for i in range(len_roads):
            roadid = roads[i][0]
            if not roadid in speeds_by_road:
                speeds_by_road[roadid] = {
                    "method1": [],
                    "method2": [],
                    "method3": [],
                    "complement": [],
                }
            speed_est = speed_roads[i]
            for key, value in speed_est.items():
                if value is not None:
                    speeds_by_road[roadid][key].append(value)

    return speeds_by_road


def speed_synthesize(speed_by_road, roadid_to_info):
    for id, speed in tqdm(speed_by_road.items(), desc="speed synthesize"):
        for key, s in speed.items():
            if s:
                s = remove_extreme(s)
                total1 = 0
                total2 = 0
                for a, b in s:
                    total1 += a * b
                    total2 += b
                speed_by_road[id][key] = total1 / total2
    level_to_speed = {
        "motorway": 14.22,
        "motorway_link": 10.60,
        "primary": 8.57,
        "unclassified": 4.70,
        "trunk": 10.70,
        "secondary": 7.37,
        "residential": 4.69,
        "trunk_link": 10.21,
        "tertiary": 6.26,
        "primary_link": 6.88,
        "living_street": 5.70,
        "secondary_link": 6.98,
        "tertiary_link": 6.72,
    }
    result = {}
    for id, speed in speed_by_road.items():
        speeds = [speed["method1"], speed["method2"], speed["method3"]]
        tmp_speed = 0
        tmp_weight = 0
        for tmp, w in zip(speeds, METHOD_WEIGHT):
            if tmp:
                tmp_speed += tmp * w
                tmp_weight += w
        if tmp_weight != 0:
            result[id] = tmp_speed / tmp_weight
        else:
            if speed["complement"]:
                s1 = speed["complement"]
                level = roadid_to_info[id]["highway"]
                if isinstance(level, str):
                    s2 = level_to_speed[level]
                else:
                    s2 = np.mean([level_to_speed[x] for x in level])
                if 0.7 * s2 < s1 < 1.3 * s2:
                    result[id] = s1
                elif 0.5 * s2 < s1 < 1.5 * s2:
                    result[id] = (s1 + s2) / 2

    return result


def speed_complement(G, speed_est_result, roadid_to_info):
    speed_est_complement = {}
    suc_cnt = 0
    for i, edge in enumerate(list(G.edges(data=True))):
        id = edge[2]["id"]
        if id in speed_est_result:
            speed_est_complement[id] = speed_est_result[id]
            continue

        Orig = edge[0]
        Dest = edge[1]
        ids_adj = []
        for edges in G._pred[Orig].values():
            for edge in edges.values():
                id_adj = edge["id"]
                if not id_adj == id:
                    ids_adj.append(id_adj)
        for edges in G._succ[Orig].values():
            for edge in edges.values():
                id_adj = edge["id"]
                if not id_adj == id:
                    ids_adj.append(id_adj)
        for edges in G._pred[Dest].values():
            for edge in edges.values():
                id_adj = edge["id"]
                if not id_adj == id:
                    ids_adj.append(id_adj)
        for edges in G._succ[Dest].values():
            for edge in edges.values():
                id_adj = edge["id"]
                if not id_adj == id:
                    ids_adj.append(id_adj)
        ids_adj = set(ids_adj)

        speeds_adj = []
        for id_adj in ids_adj:
            if id_adj in speed_est_result:
                speeds_adj.append(speed_est_result[id_adj])
        if not speeds_adj == []:
            speed_est_complement[id] = np.mean(speeds_adj)
            suc_cnt = suc_cnt + 1

    level_to_speed = defaultdict(list)
    to_be_complement = []
    for edge_id, info in roadid_to_info.items():
        level = info["highway"]
        if edge_id in speed_est_complement:
            if isinstance(level, str):
                level_to_speed[level].append(speed_est_complement[edge_id])
            else:
                for t in level:
                    level_to_speed[t].append(speed_est_complement[edge_id])
        else:
            to_be_complement.append((edge_id, level))
    level_to_speed = {
        level: np.mean(speeds) for level, speeds in level_to_speed.items()
    }
    for edge_id, level in to_be_complement:
        if isinstance(level, str):
            speed_est_complement[edge_id] = level_to_speed[level]
        else:
            speed_est_complement[edge_id] = np.mean([level_to_speed[t] for t in level])

    print("estimated average speed(km/h) for each road level:")
    t = sorted(list(level_to_speed.items()), key=lambda x: x[1])
    for level, speed in t:
        print(level, " " * (20 - len(level)), round(speed * 3.6, 2))

    print(
        "final average speed(km/h):", 3.6 * np.mean(list(speed_est_complement.values()))
    )
    return speed_est_complement


if __name__ == "__main__":
    G = pickle.load(open(road_path, "rb"))
    roadid_to_info = {}
    for u, v, k in G.edges:
        edge_info = G.edges[u, v, k]
        roadid_to_info[edge_info["id"]] = edge_info

    traj_match = pickle.load(open(matched_traj_path, "rb"))
    traj_match = preprocess(traj_match, roadid_to_info)

    trajs_for_each_slice = defaultdict(list)
    ave_speed_for_each_slice = []
    speed_est_for_each_slice = []
    speed_est_simple_for_each_slice = []
    for traj in traj_match:
        start_tm = int(traj[0][1][0][1] / 3600)
        assert 0 <= start_tm < 24
        trajs_for_each_slice[start_tm].append(traj)
    for start_tm in range(24):
        traj_match = trajs_for_each_slice[start_tm]
        print(f"------------   start_hour:{start_tm}   ------------")
        print("input traj num:", len(traj_match))
        speeds_by_road = speed_estimate(traj_match, roadid_to_info)
        speed_est = speed_synthesize(speeds_by_road, roadid_to_info)
        ave_speed_for_each_slice.append(3.6 * np.mean(list(speed_est.values())))
        speed_est_for_each_slice.append(deepcopy(speed_est))
        speed_est = speed_complement(G, speed_est, roadid_to_info)
        speed_est_simple_for_each_slice.append(deepcopy(speed_est))
    print([len(x) for x in speed_est_for_each_slice])
    print([len(x) for x in speed_est_simple_for_each_slice])
    pickle.dump(speed_est_for_each_slice, open("data/road_speed_slice.pkl", "wb"))
    pickle.dump(
        speed_est_simple_for_each_slice, open("data/road_speed_simple_slice.pkl", "wb")
    )
    pickle.dump(
        speed_est_simple_for_each_slice,
        open("../dataset/road_speed_simple_slice.pkl", "wb"),
    )
