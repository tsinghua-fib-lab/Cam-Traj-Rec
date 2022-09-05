"""
the clustering algorithm
"""
from tqdm import tqdm
import coloredlogs
import logging
import numpy as np
import faiss
from collections import defaultdict

coloredlogs.install(fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s")


def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)


class FlatSearcher:
    def __init__(self, ngpu=1, feat_len=256):
        if ngpu:
            flat_config = []
            for i in range(ngpu):
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = False
                cfg.device = i
                flat_config.append(cfg)
            res = [faiss.StandardGpuResources() for _ in range(ngpu)]
            indexes = [
                faiss.GpuIndexFlatIP(res[i], feat_len, flat_config[i])
                for i in range(ngpu)
            ]
            self.index = faiss.IndexProxy()
            for sub_index in indexes:
                self.index.addIndex(sub_index)
        else:
            self.index = faiss.IndexFlatL2(feat_len)

    def search_by_topk(self, query, gallery, topk=16):
        self.index.reset()
        self.index.add(gallery)
        topk_scores, topk_idxs = self.index.search(query, topk)
        return topk_scores, topk_idxs


class Cluster:
    def __init__(self, ngpu=1, similarity_threshold=0.88, topK=128):
        self.records = []
        self.clusters = []
        self.car_features = []
        self.plate_features = []
        self.plate_true_index = []

        self.similarity_threshold = similarity_threshold
        self.topK = topK
        self.reid_searcher = FlatSearcher(feat_len=256, ngpu=ngpu)
        self.plate_searcher = FlatSearcher(feat_len=256, ngpu=ngpu)

    def rough_search(self, records):
        reid_gallerys = []
        plate_gallerys = []
        plate_texts = []
        for record in records:
            reid_gallerys.append(record["car_feature"])
            if record["plate_feature"] is not None:
                plate_gallerys.append(record["plate_feature"])
            if record["plate_text"] is not None:
                plate_texts.append(record["plate_text"])

        reid_gallerys = np.array(reid_gallerys)
        plate_gallerys = np.array(plate_gallerys)
        self.reid_topk_idxs = self.reid_searcher.search_by_topk(
            reid_gallerys, reid_gallerys, topk=self.topK
        )[1].tolist()
        self.plate_topk_idxs = self.plate_searcher.search_by_topk(
            plate_gallerys, plate_gallerys, topk=self.topK
        )[1].tolist()

    def add_cluster(self, record):
        record["cluster_id"] = len(self.clusters)
        cluster = {
            "car_feature": np.copy(record["car_feature"]),
            "car_feature_sum": [np.copy(record["car_feature"])],
            "plate_feature": None,
            "plate_feature_sum": [],
        }
        if record["plate_feature"] is not None:
            cluster["plate_feature"] = np.copy(record["plate_feature"])
            cluster["plate_feature_sum"] = [np.copy(record["plate_feature"])]
        self.clusters.append(cluster)

    def merge_cluster(self, record, i):
        record["cluster_id"] = i
        cluster = self.clusters[i]
        cluster["car_feature_sum"].append(record["car_feature"])
        cluster["car_feature"] = normalize(np.mean(cluster["car_feature_sum"], axis=0))
        if record["plate_feature"] is not None:
            cluster["plate_feature_sum"].append(record["plate_feature"])
            cluster["plate_feature"] = normalize(
                np.mean(cluster["plate_feature_sum"], axis=0)
            )

    def similarity(self, record1, record2, k=0.1):
        reid_score = record1["car_feature"] @ record2["car_feature"]
        if (
            record1["plate_feature"] is not None
            and record2["plate_feature"] is not None
        ):
            plate_score = record1["plate_feature"] @ record2["plate_feature"]
            return k * reid_score + (1 - k) * plate_score
        else:
            return reid_score

    def add(self, record):
        record["car_feature"] = normalize(record["car_feature"])
        if record["plate_feature"] is not None:
            record["plate_feature"] = normalize(record["plate_feature"])
        self.records.append(record)

        reid_num = len(self.car_features)
        plate_num = len(self.plate_features)
        record_index = len(self.records) - 1
        self.car_features.append(record["car_feature"])
        if record["plate_feature"] is not None:
            self.plate_features.append(record["plate_feature"])
            self.plate_true_index.append(record_index)

        if record_index == 0:
            self.add_cluster(record)
        else:
            recall_indexs = [x for x in self.reid_topk_idxs[reid_num] if x < reid_num]
            if record["plate_feature"] is not None and plate_num > 0:
                recall_indexs += [
                    self.plate_true_index[x]
                    for x in self.plate_topk_idxs[plate_num]
                    if x < plate_num
                ]

            max_similarity = -1
            bestId = 0
            visited_cids = set()
            cids = {self.records[k]["cluster_id"] for k in set(recall_indexs)}

            for cluster_id in cids:
                similarity = self.similarity(record, self.clusters[cluster_id])
                if similarity > max_similarity:
                    max_similarity = similarity
                    bestId = cluster_id
                visited_cids.add(cluster_id)
            if max_similarity >= self.similarity_threshold:
                self.merge_cluster(record, bestId)
            else:
                self.add_cluster(record)

    def get_records_and_labels(self):
        return self.records, [i["cluster_id"] for i in self.records]


class SigCluster:
    def __init__(self, feature_dims=[256, 256], ngpu=1):
        self.searchers = {i: FlatSearcher(ngpu, i) for i in set(feature_dims)}
        self.f_dims = feature_dims

    def fit(
        self,
        data,
        initial_labels=None,
        weights=[0.1, 0.9],
        similarity_threshold=0.88,
        topK=128,
        normalized=True,
    ):
        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights] * len(self.f_dims)
        else:
            assert len(weights) == len(self.f_dims)
        if isinstance(topK, int):
            topK = [topK] * len(self.f_dims)
        else:
            assert len(topK) == len(self.f_dims)

        N = len(data)
        N_f = len(data[0])

        if normalized:
            logging.info("Normalize")
            data = [
                [None if j is None else normalize(j) for j in i] for i in tqdm(data)
            ]

        logging.info("Search topk")

        data_ = list(zip(*data))
        fs = []
        f_ids = []
        for i in data_:
            f_id, f = zip(*((j, k) for j, k in enumerate(i) if k is not None))
            fs.append(np.array(f))
            f_ids.append(f_id)

        f_topks = [
            [
                [f_id[k] for k in j if k < i]
                for i, j in enumerate(
                    self.searchers[dim].search_by_topk(f, f, topk)[1].tolist()
                )
            ]
            for f, f_id, dim, topk in zip(fs, f_ids, self.f_dims, topK)
        ]
        assert all(len(i[0]) == 0 for i in f_topks)

        topks = [[] for _ in range(len(data))]
        for f_topk, f_id in zip(f_topks, f_ids):
            for i, topk in zip(f_id, f_topk):
                topks[i] += topk

        if not normalized:
            data = [
                [None if j is None else normalize(j) for j in i] for i in tqdm(data)
            ]

        logging.info("Clustering")
        cf_means = {}
        cfs = {}
        if initial_labels is None:
            cids = [-1] * N
        else:
            cids = initial_labels
            cid2records = defaultdict(list)
            for cid, record in zip(cids, data):
                if cid >= 0:
                    cid2records[cid].append(record)
            for cid, rs in cid2records.items():
                tmp = cfs[cid] = [[j for j in i if j is not None] for i in zip(*rs)]
                cf_means[cid] = [
                    normalize(np.mean(t, axis=0)) if len(t) else None for t in tmp
                ]
        for i, (record, topk) in enumerate(zip(tqdm(data), topks)):
            if cids[i] >= 0:
                continue
            cs = {cids[i] for i in topk}
            best_cid = -1
            best_sim = -1
            for c in cs:
                w_total = 0
                sim = 0
                for w, a, b in zip(weights, record, cf_means[c]):
                    if a is not None and b is not None:
                        sim += a @ b * w
                        w_total += w
                sim /= w_total
                if sim > best_sim:
                    best_sim = sim
                    best_cid = c
            if best_cid >= 0 and best_sim >= similarity_threshold:
                cids[i] = best_cid
                cf = cfs[best_cid]
                cf_mean = cf_means[best_cid]
                for j, k in enumerate(record):
                    if k is not None:
                        if cf[j]:
                            cf[j].append(k)
                            cf_mean[j] = normalize(np.mean(cf[j], axis=0))
                        else:
                            cf[j] = [k]
                            cf_mean[j] = k
            else:
                cid = len(cf_means)
                cids[i] = cid
                cf_means[cid] = record
                cfs[cid] = [[] if j is None else [] for j in record]
        logging.info("done")
        return cids
