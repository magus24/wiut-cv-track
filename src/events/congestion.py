"""congestion event detector (PHASE 18).

Definition: a stable accumulation of VEHICLES in a bounded part of the road,
with substantially reduced motion / a large stopped share of the traffic.
"congestion" is NOT vehicle_count >= N: a dense but normally flowing stream is
NOT congestion. The detector aggregates three independent signals per spatial
cluster and only then reports congestion:

    valid vehicles -> spatial cluster (lane-aware) -> sufficient density
                      + low motion + stationary/slow ratio temporal persistence
                      -> TemporalEventEngine -> "congestion" event

DISTINCTIONS BUILT INTO THE LOGIC
  A. NORMAL TRAFFIC    : many vehicles, most moving at normal speed
                         -> median/mean speed above the caps OR stationary
                            ratio below the floor -> NOT congestion.
  B. TEMPORARY STOP    : a few vehicles stop briefly (light, stop line,
                         pedestrian, turning) -> below `min_vehicle_count` /
                         below the temporal confirmation -> NOT congestion.
  C. CONGESTION        : several vehicles in one local road area, speed
                         substantially reduced, meaningful stationary/slow
                         share, persisting -> congestion.

NOT depending on: red_light, stop_line, traffic-light state. The detector
never calls get_traffic_light_state() (a spy in the unit tests hard-fails if it
does), so "signal" is always None and UNKNOWN/RED/GREEN are all irrelevant.

SPATIAL CLUSTERING (deterministic, no ML)
  Priority: 1) configured lane (geometry.get_lane) 2) road zone 3) fallback
  spatial clustering. Vehicles are partitioned by lane membership first:
  congestion in one lane never merges with a spatially separated lane.
  Inside one partition, a distance-threshold union-find (grid-binned so it is
  near-linear, no O(N^2) pair scan) builds connected components with
  `cluster_distance_px`. A cluster is rejected if its spatial extent
  (bounding-box diagonal) exceeds `max_cluster_extent_px`.

CLUSTER IDENTITY (bounded, per cluster)
  Each tracked cluster keeps: cluster_id / centroid / first_seen / last_seen /
  last_congestion_time / active / its OWN TemporalEventEngine instance. A new
  frame's cluster is assigned to a tracked cluster when the centroids are
  within `cluster_track_dist_px` OR they share >= `min_cluster_overlap` vehicle
  ids -> membership changes on +/-1 vehicle do not create a new episode.
  Unmatched tracked clusters are aged out after `max_cluster_gap_sec` (their
  engine is flushed, segments preserved). There is NO state inheritance between
  unrelated clusters.

MOTION (never recomputed)
  Per-vehicle motion comes exclusively from the shared MotionEngine results in
  `motion_states` (speed / stationary / accel / quality). Classifications:
      stationary:  st.stationary or speed <= stationary_speed_px_s
      slow:        speed <= slow_speed_px_s
  Low-quality tracks (quality < `min_quality`) are excluded from cluster
  statistics entirely (counted separately, `excluded` in the report).

CONGESTION EVIDENCE (per cluster, all of the following)
  count        >= min_vehicle_count
  ( branch A: stationary      >= min_stationary_count AND
              stationary_ratio >= min_stationary_ratio
    OR branch B: slow_ratio   >= min_slow_ratio )
  median_speed <= max_median_speed_px_s
  mean_speed   <= max_mean_speed_px_s
  extent       <= max_cluster_extent_px
  density      >= min_cluster_density   (>= 1.0 to enable; 0.0 disables)

Branches A/B let BOTH a stopped queue (a few stationary cars) and a slow crawl
(all cars creeping, none fully stopped) count as congestion — a crawling jam is
NOT normal traffic. A single stationary vehicle among fast cars violates both
branches (stationary count/ratio too low, slow ratio too low).

STRICTLY CAUSAL: per frame only t <= current time is used (no future vehicle
positions / speeds / cluster membership; no post-hoc trajectory). Every frame
the detector (1) builds the valid vehicle set, (2) builds clusters, (3) scores
each cluster, (4) matches identity and feeds the per-cluster TemporalEventEngine
with evidence, so TWO distant congested clusters at the same time are TWO
independent episodes. Deterministic: fixed seeds not needed, no randomness.

ROAD GEOMETRY FALLBACK
  When `geometry` provides a road polygon the on-road test is applied and
  out-of-road vehicles are excluded. When geometry/road polygon is absent the
  detector documents and uses the spatial-clustering-only fallback (road check
  skipped, clustering still applied) -> no new road polygon is ever created.

Per-cluster reject reasons (ordered, first failing gate wins):
  insufficient_vehicles, insufficient_stationary, insufficient_stationary_ratio,
  insufficient_slow_ratio, median_speed_high, mean_speed_high,
  extent_too_large, low_density.
"""

from __future__ import annotations

import math
import statistics

from .temporal import EventSegment, TemporalEventEngine

LABEL = "congestion"
DEFAULT_VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")

# canonical ordered rejection reasons (CSV / debug reports order-friendly)
REASONS = ("insufficient_vehicles", "insufficient_stationary",
           "insufficient_stationary_ratio", "insufficient_slow_ratio",
           "median_speed_high", "mean_speed_high", "extent_too_large",
           "low_density")


class CongestionDetector:
    """Per-frame congestion evidence for the shared scene.

    Signature mirrors the other detectors: `update(tracks, motion_states,
    geometry, t_sec)` -> report; `finalize()` -> confirmed "congestion"
    segments; `reset()`. The temporal engine is reused: each tracked cluster
    owns one TemporalEventEngine instance (label "congestion"), so clusters
    never merge and simultaneous distant jams stay separate episodes.
    """

    def __init__(
        self,
        vehicle_labels: tuple = DEFAULT_VEHICLE_LABELS,
        min_vehicle_count: int = 4,
        min_stationary_count: int = 2,
        min_stationary_ratio: float = 0.5,
        min_slow_ratio: float = 0.6,
        stationary_speed_px_s: float = 8.0,
        slow_speed_px_s: float = 25.0,
        max_median_speed_px_s: float = 35.0,
        max_mean_speed_px_s: float = 45.0,
        cluster_distance_px: float = 120.0,
        max_cluster_extent_px: float = 400.0,
        min_cluster_density: float = 0.0,
        min_quality: float = 0.3,
        max_cluster_gap_sec: float = 2.0,
        min_cluster_overlap: int = 2,
        cluster_track_dist_px: float | None = None,
        min_on_duration: float = 1.0,
        allowed_gap: float = 1.0,
        merge_gap: float = 2.0,
        min_duration: float = 1.0,
    ) -> None:
        assert min_vehicle_count >= 1
        assert min_stationary_count >= 0
        assert 0.0 <= min_stationary_ratio <= 1.0
        assert 0.0 <= min_slow_ratio <= 1.0
        assert stationary_speed_px_s >= 0.0
        assert slow_speed_px_s >= stationary_speed_px_s
        assert max_median_speed_px_s > 0.0
        assert max_mean_speed_px_s > 0.0
        assert cluster_distance_px > 0.0
        assert max_cluster_extent_px > 0.0
        assert min_cluster_density >= 0.0
        assert max_cluster_gap_sec > 0.0
        assert min_cluster_overlap >= 0
        self.vehicle_labels = frozenset(vehicle_labels)
        self.min_vehicle_count = int(min_vehicle_count)
        self.min_stationary_count = int(min_stationary_count)
        self.min_stationary_ratio = float(min_stationary_ratio)
        self.min_slow_ratio = float(min_slow_ratio)
        self.stationary_speed = float(stationary_speed_px_s)
        self.slow_speed = float(slow_speed_px_s)
        self.max_median_speed = float(max_median_speed_px_s)
        self.max_mean_speed = float(max_mean_speed_px_s)
        self.cluster_distance = float(cluster_distance_px)
        self.max_extent = float(max_cluster_extent_px)
        self.min_density = float(min_cluster_density)
        self.min_quality = float(min_quality)
        self.max_cluster_gap = float(max_cluster_gap_sec)
        self.min_overlap = int(min_cluster_overlap)
        self.track_dist = (float(cluster_track_dist_px)
                           if cluster_track_dist_px is not None
                           else self.cluster_distance * 1.5)
        self.candidate_labels = self.vehicle_labels
        self._temp = dict(min_on_duration=min_on_duration,
                          allowed_gap=allowed_gap,
                          merge_gap=merge_gap, min_duration=min_duration)
        # bounded causal state (per tracked cluster)
        self._clusters: dict[int, dict] = {}
        self._next_cid: int = 0
        self._stats: dict[int, dict] = {}       # cumulative per-cluster stats
        self._segments_map: dict[int, list[EventSegment]] = {}

    # ------------------------------------------------------------------ API
    def update(self, tracks, motion_states, geometry=None, t_sec: float = 0.0) -> dict:
        """Evaluate congestion evidence at time t_sec (strictly causal).

        Args:
            tracks:        dict {track_id -> TrackTrajectory} (position + label).
            motion_states: dict {track_id -> MotionState} (smoothed motion).
            geometry:      Geometry (road/lane polygons; optional fallback).
            t_sec:         current time.
        Returns per-frame report:
            {"t_sec", "evidence", "valid_vehicle_count", "excluded",
             "vehicles": {tid: record}, "clusters": {cid: stats},
             "active_clusters": [cids], "rejected": {cid: reason},
             "road_available", "lane_available", "signal": None}
        and feeds ("congestion", evidence) into each tracked cluster's engine.
        """
        road_available = geometry is not None and bool(geometry.road_polygon)
        lane_available = geometry is not None and bool(geometry.lanes)

        valid, excluded = self._collect(tracks, motion_states, geometry,
                                        road_available, lane_available)

        # partition by lane (priority 1) -> spatial clusters inside a lane
        groups: dict[str, list] = {}
        for v in valid:
            groups.setdefault(v["lane_id"] or "<none>", []).append(v)
        raw: list[dict] = []
        for gkey in sorted(groups):
            raw.extend(self._cluster_group(groups[gkey]))

        used: dict[int, dict] = {}
        active_clusters: list[int] = []
        rejected: dict[int, str] = {}
        vehicles_records: dict[int, dict] = {v["tid"]: {} for v in valid}
        clusters_report: dict[int, dict] = {}

        # deterministic assignment: new clusters sorted, tracked cids ascending
        owned = set()
        for c in raw:
            cid = self._match_tracked(c, owned)
            owned.add(cid)
            if cid not in self._clusters:
                self._clusters[cid] = {
                    "centroid": c["centroid"], "first_seen": t_sec,
                    "last_seen": t_sec, "last_congestion": None,
                    "lane_id": c["lane_id"],
                    "members_last": frozenset(c["tids"]),
                    "engine": TemporalEventEngine(**self._temp),
                }
            cs = self._clusters[cid]
            cs["centroid"] = c["centroid"]
            cs["last_seen"] = t_sec
            cs["lane_id"] = c["lane_id"]
            cs["members_last"] = frozenset(c["tids"])
            used[cid] = c

            self._update_stats(cid, c, excluded, t_sec)
            clusters_report[cid] = dict(c, cluster_id=cid)
            if c["evidence"]:
                active_clusters.append(cid)
                cs["last_congestion"] = t_sec
            else:
                rejected[cid] = c["reason"]
            cs["engine"].update(LABEL, t_sec, evidence=c["evidence"])

            for tid in c["tids"]:
                vehicles_records[tid] = {
                    "cluster_id": cid, "lane_id": c["lane_id"],
                    "stationary": c["by_id"][tid]["stationary"],
                    "slow": c["by_id"][tid]["slow"],
                    "speed": c["by_id"][tid]["speed"],
                    "quality": c["by_id"][tid]["quality"],
                    "cls": c["by_id"][tid]["cls"], "reason": None,
                }

        # unmatched tracked clusters still alive this frame: report absence
        for cid in sorted(self._clusters):
            if cid not in used:
                self._clusters[cid]["engine"]\
                    .update(LABEL, t_sec, evidence=False)

        self._prune(t_sec)

        active_clusters.sort()
        evidence = bool(active_clusters)
        return {
            "t_sec": t_sec, "evidence": evidence,
            "valid_vehicle_count": len(valid),
            "excluded": excluded,
            "vehicles": {tid: (vehicles_records.get(tid) or
                               {"cluster_id": None, "lane_id": None,
                                "stationary": None, "slow": None,
                                "speed": None, "quality": None,
                                "cls": None, "reason": None})
                         for tid in tracks},
            "clusters": clusters_report,
            "active_clusters": active_clusters,
            "rejected": rejected,
            "road_available": road_available,
            "lane_available": lane_available,
            "signal": None,
        }

    def finalize(self) -> list[EventSegment]:
        """Close every still-active cluster engine and return the confirmed
        "congestion" segments (one episode per tracked cluster), sorted."""
        for cid in list(self._clusters):
            self._flush(cid)
        out: list[EventSegment] = []
        for segs in self._segments_map.values():
            out.extend(segs)
        out.sort(key=lambda s: (s.label, s.start, s.end))
        return out

    def reset(self) -> None:
        """Clear all cluster state so the detector can start a new video."""
        for cid in list(self._clusters):
            self._flush(cid)
        self._clusters.clear()
        self._stats.clear()
        self._segments_map.clear()
        self._next_cid = 0

    # --------------------------------------------------------------- collect
    def _collect(self, tracks, motion_states, geometry, road_available,
                 lane_available):
        excluded = {"not_vehicle": [], "no_motion_state": [], "low_quality": [],
                    "outside_road": []}
        valid: list[dict] = []
        for tid, tr in tracks.items():
            if tr.last is None:
                continue
            if tr.label not in self.vehicle_labels:
                excluded["not_vehicle"].append(tid)
                continue
            st = motion_states.get(tid)
            if st is None:
                excluded["no_motion_state"].append(tid)
                continue
            if st.quality < self.min_quality:
                excluded["low_quality"].append(tid)
                continue
            pos = (tr.last.x, tr.last.bottom_y)
            if road_available and not geometry.is_on_road(pos):
                excluded["outside_road"].append(tid)
                continue
            lane_id = geometry.get_lane(pos) if lane_available else None
            speed = st.speed
            valid.append({
                "tid": tid, "pos": pos, "speed": speed,
                "stationary": bool(st.stationary or speed <= self.stationary_speed),
                "slow": bool(speed <= self.slow_speed),
                "quality": st.quality, "lane_id": lane_id, "cls": tr.label,
            })
        valid.sort(key=lambda v: (v["pos"][0], v["pos"][1], v["tid"]))
        return valid, excluded

    # -------------------------------------------------------------- cluster
    def _cluster_group(self, members: list[dict]) -> list[dict]:
        """Distance-threshold connected components (grid-binned union-find).
        Deterministic: members arrive sorted; roots tie-broken by smallest idx."""
        if not members:
            return []
        n = len(members)
        cell = max(self.cluster_distance, 1.0)
        bins: dict[tuple[int, int], list[int]] = {}
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i, m in enumerate(members):
            gx, gy = int(m["pos"][0] // cell), int(m["pos"][1] // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in bins.get((gx + dx, gy + dy), ()):
                        if i != j and self._dist2(m["pos"], members[j]["pos"]) \
                                <= self.cluster_distance ** 2:
                            union(i, j)
            bins.setdefault((gx, gy), []).append(i)

        roots: dict[int, list[int]] = {}
        for i in range(n):
            roots.setdefault(find(i), []).append(i)
        clusters = []
        for root in sorted(roots):
            member_list = [members[i] for i in sorted(roots[root])]
            clusters.append(self._score_cluster(member_list))
        clusters.sort(key=lambda c: (c["centroid"][0], c["centroid"][1]))
        return clusters

    def _score_cluster(self, members: list[dict]) -> dict:
        n = len(members)
        speeds = [m["speed"] for m in members]
        stationary = sum(1 for m in members if m["stationary"])
        slow = sum(1 for m in members if m["slow"])
        xs = [m["pos"][0] for m in members]
        ys = [m["pos"][1] for m in members]
        extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        area = (max(xs) - min(xs) + 1.0) * (max(ys) - min(ys) + 1.0)
        density = n / area
        median = float(statistics.median(speeds))
        mean = sum(speeds) / n
        c = {
            "cluster_id": None, "centroid": (sum(xs) / n, sum(ys) / n),
            "lane_id": members[0]["lane_id"],
            "count": n,
            "moving": n - slow, "slow": slow, "stationary": stationary,
            "stationary_ratio": stationary / n, "slow_ratio": slow / n,
            "median_speed": median, "mean_speed": mean,
            "extent": extent, "density": density,
            "tids": [m["tid"] for m in members],
            "by_id": {m["tid"]: m for m in members},
        }
        reason = self._reject_reason(c)
        c["evidence"] = reason is None
        c["reason"] = reason
        return c

    def _reject_reason(self, c: dict) -> str | None:
        if c["count"] < self.min_vehicle_count:
            return "insufficient_vehicles"
        branch_a = (c["stationary"] >= self.min_stationary_count and
                    c["stationary_ratio"] >= self.min_stationary_ratio)
        branch_b = c["slow_ratio"] >= self.min_slow_ratio
        if not (branch_a or branch_b):
            if c["stationary"] < self.min_stationary_count:
                return "insufficient_stationary"
            if c["stationary_ratio"] < self.min_stationary_ratio:
                return "insufficient_stationary_ratio"
            return "insufficient_slow_ratio"
        if c["median_speed"] > self.max_median_speed:
            return "median_speed_high"
        if c["mean_speed"] > self.max_mean_speed:
            return "mean_speed_high"
        if c["extent"] > self.max_extent:
            return "extent_too_large"
        if self.min_density > 0.0 and c["density"] < self.min_density:
            return "low_density"
        return None

    # ---------------------------------------------------------------- match
    def _match_tracked(self, c: dict, owned: set) -> int:
        """Assign a fresh cluster to the best tracked identity (or spawn one)."""
        best = None
        best_key = None
        for cid in sorted(self._clusters):
            if cid in owned:
                continue
            cs = self._clusters[cid]
            shared = len(set(c["tids"]) & set(cs["members_last"]))
            d = self._dist(c["centroid"], cs["centroid"])
            hit = shared >= self.min_overlap or d <= self.track_dist
            if not hit:
                continue
            key = (shared, -d, -cid)     # overlap first, then proximity
            if best_key is None or key > best_key:
                best, best_key = cid, key
        if best is None:
            best = self._next_cid
            self._next_cid += 1
        return best

    # ---------------------------------------------------------------- stats
    def _update_stats(self, cid: int, c: dict, excluded, t_sec: float) -> None:
        st = self._stats.get(cid)
        if st is None:
            st = {"cluster_id": cid, "lane_id": c["lane_id"],
                  "first_seen": t_sec, "last_seen": t_sec,
                  "max_count": c["count"], "min_median_speed": c["median_speed"],
                  "max_stationary_ratio": c["stationary_ratio"],
                  "max_slow_ratio": c["slow_ratio"],
                  "max_extent": c["extent"], "density": c["density"],
                  "members": set(c["tids"]), "observed_frames": 0,
                  "evidence_frames": 0, "reject_reason": None}
            self._stats[cid] = st
        st["last_seen"] = t_sec
        st["max_count"] = max(st["max_count"], c["count"])
        st["min_median_speed"] = min(st["min_median_speed"], c["median_speed"])
        st["max_stationary_ratio"] = max(st["max_stationary_ratio"],
                                         c["stationary_ratio"])
        st["max_slow_ratio"] = max(st["max_slow_ratio"], c["slow_ratio"])
        st["max_extent"] = max(st["max_extent"], c["extent"])
        st["members"].update(c["tids"])
        st["observed_frames"] += 1
        if c["evidence"]:
            st["evidence_frames"] += 1
        elif st["reject_reason"] is None:
            st["reject_reason"] = c["reason"]

    def _flush(self, cid: int) -> None:
        cs = self._clusters.pop(cid, None)
        if cs is not None:
            segs = [s for s in cs["engine"].finalize() if s.end > s.start]
            self._segments_map.setdefault(cid, []).extend(segs)

    # ---------------------------------------------------------------- prune
    def _prune(self, t_sec: float) -> None:
        stale = [cid for cid, cs in self._clusters.items()
                 if t_sec - cs["last_seen"] > self.max_cluster_gap]
        for cid in stale:
            self._flush(cid)

    # ------------------------------------------------------------------ util
    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _dist2(a, b) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy


def segments_to_events(segments: list[EventSegment]) -> list[list]:
    """EventSegments -> harness events [[start, end, label]]. (Part A glue.)"""
    return [s.to_list() for s in segments]