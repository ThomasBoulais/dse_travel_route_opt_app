import numpy as np
from typing import List, Dict, Set, Tuple


class TDTOPTWEnv:
    def __init__(
        self,
        poi_features: np.ndarray,
        poi_scores: np.ndarray,
        opening_mask: np.ndarray,
        travel_time: np.ndarray,
        knn_neighbors: List[List[int]],
        main_categories: np.ndarray,
        is_accommodation: np.ndarray,
        visit_durations: np.ndarray,
        start_poi_idx: int,
        start_day: int,
        day_start_minute: int,
        day_end_minute: int,
        num_days: int,
        max_steps: int,
        reward_cfg: Dict,
    ):
        self.poi_features = poi_features
        self.poi_scores = poi_scores
        self.opening_mask = opening_mask
        self.travel_time = travel_time

        # KNN neighbors: list of lists, neighbors[i] = [j1, j2, ...]
        self.knn_neighbors = knn_neighbors
        self.max_actions = max(len(n) for n in knn_neighbors)

        self.main_categories = main_categories
        self.is_accommodation = is_accommodation.astype(bool)
        self.visit_durations = visit_durations.astype(np.float32)

        self.N = poi_features.shape[0]
        self.d = poi_features.shape[1]

        # Keep both names for compatibility with training code
        self.start_poi_idx = start_poi_idx
        self.start_poi = start_poi_idx
        self.start_day = start_day
        self.day_start_minute = day_start_minute
        self.day_end_minute = day_end_minute
        self.num_days = num_days
        self.max_steps = max_steps

        self.total_time_budget = num_days * (day_end_minute - day_start_minute)
        self.reward_cfg = reward_cfg

        self.reset()

    # --- time helpers ---
    def _time_to_indices(self, delta_min: float):
        total = self.current_day * 1440 + self.current_minute + int(delta_min)
        day = (total // 1440) % 7
        minute = total % 1440
        return day, minute

    def _is_open(self, poi_idx: int, day: int, minute: int) -> bool:
        return self.opening_mask[poi_idx, day, minute] == 1

    def _day_index_from_start(self, day: int) -> int:
        return (day - self.start_day) % 7

    # --- RL API ---
    def reset(self):
        self.current_poi = self.start_poi
        self.current_day = self.start_day
        self.current_minute = self.day_start_minute
        self.steps = 0

        self.visited = np.zeros(self.N, dtype=bool)
        self.visited[self.current_poi] = True

        self.used_time = 0.0
        self.visited_categories: Set[str] = {self.main_categories[self.current_poi]}

        self.accommodation_used_today = False
        self.last_day = self.start_day

        self.lunch_taken_today = False

        return self._get_state()

    def _get_state(self):
        feat = self.poi_features[self.current_poi]
        time_ratio = np.array(
            [self.used_time / max(self.total_time_budget, 1e-6)], dtype=np.float32
        )
        day_norm = np.array([self.current_day / 6.0], dtype=np.float32)
        minute_norm = np.array([self.current_minute / 1440.0], dtype=np.float32)
        visited_ratio = np.array([self.visited.sum() / self.N], dtype=np.float32)
        diversity_ratio = np.array(
            [len(self.visited_categories) / max(self.N, 1)], dtype=np.float32
        )
        return np.concatenate(
            [feat, time_ratio, day_norm, minute_norm, visited_ratio, diversity_ratio]
        ).astype(np.float32)

    def _feasible_actions_mask(self) -> np.ndarray:
        neighbors = self.knn_neighbors[self.current_poi]
        K = len(neighbors)

        mask = np.zeros(K, dtype=bool)

        for a in range(K):
            j = neighbors[a]

            if self.visited[j]:
                continue

            tt = self.travel_time[self.current_poi, j]
            if np.isinf(tt):
                continue

            visit_dur = self.visit_durations[j]
            total = tt + visit_dur

            day_arr, min_arr = self._time_to_indices(tt)
            day_dep, min_dep = self._time_to_indices(total)
            day_idx = self._day_index_from_start(day_dep)
            if day_idx >= self.num_days:
                continue

            if not self._is_open(j, day_arr, min_arr):
                continue

            if min_dep > self.day_end_minute + 60 and not self.is_accommodation[j]:
                continue

            if self.is_accommodation[j] and (
                self.accommodation_used_today or self.current_minute < 18 * 60
            ):
                continue

            if self.main_categories[j] == "restauration":
                if 660 <= self.current_minute <= 900 and self.lunch_taken_today:
                    continue

            mask[a] = True

        # PAD to max_actions
        padded = np.zeros(self.max_actions, dtype=bool)
        padded[:K] = mask
        return padded

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        feas_mask = self._feasible_actions_mask()

        if action_idx is None or action_idx < 0 or action_idx >= len(feas_mask):
            return self._get_state(), self.reward_cfg["invalid_penalty"], True, {
                "reason": "invalid_action"
            }

        if not feas_mask[action_idx]:
            return self._get_state(), self.reward_cfg["invalid_penalty"], True, {
                "reason": "infeasible"
            }

        prev_poi = self.current_poi
        neighbors = self.knn_neighbors[self.current_poi]
        if action_idx >= len(neighbors):
            return self._get_state(), self.reward_cfg["invalid_penalty"], True, {
                "reason": "invalid_action"
            }
        next_poi = neighbors[action_idx]

        tt = self.travel_time[prev_poi, next_poi]
        visit_dur = self.visit_durations[next_poi]
        total = tt + visit_dur

        if self.current_day != self.last_day:
            self.accommodation_used_today = False
            self.lunch_taken_today = False
            self.last_day = self.current_day

        if self.is_accommodation[next_poi]:
            self.accommodation_used_today = True

        if 660 <= self.current_minute <= 900 and self.main_categories[next_poi] == "restauration":
            self.lunch_taken_today = True

        day_dep, min_dep = self._time_to_indices(total)
        day_idx = self._day_index_from_start(day_dep)

        self.used_time += total
        self.current_day, self.current_minute = day_dep, min_dep
        self.current_poi = next_poi
        self.visited[next_poi] = True
        self.steps += 1

        cat = self.main_categories[next_poi]
        new_category = cat not in self.visited_categories
        if new_category:
            self.visited_categories.add(cat)

        r_cfg = self.reward_cfg
        interest = float(self.poi_scores[next_poi])
        travel_penalty = r_cfg["travel_penalty"] * tt

        prev_coord = self.poi_features[prev_poi][0:2]
        next_coord = self.poi_features[next_poi][0:2]
        dist = float(np.linalg.norm(prev_coord - next_coord))
        distance_penalty = r_cfg["distance_penalty"] if dist < 0.001 else 0.0

        diversity_bonus = r_cfg["diversity_bonus"] if new_category else 0.0
        time_usage_bonus = r_cfg["time_usage_bonus"] * total
        step_bonus = r_cfg["step_bonus"]

        reward = (
            interest
            + travel_penalty
            + distance_penalty
            + diversity_bonus
            + time_usage_bonus
            + step_bonus
        )

        visited_count = self.visited.sum()
        reward += 0.5 * visited_count

        if dist > 0.01:
            reward += 0.8

        if self.current_minute > 12 * 60:
            reward += 0.8

        if self.is_accommodation[self.current_poi] and self.current_minute > 18 * 60:
            reward += 2.0

        done = False
        info: Dict = {}

        if day_idx >= self.num_days:
            done = True
            info["reason"] = "horizon_exceeded"

        if (
            self.current_minute > self.day_end_minute
            and not self.is_accommodation[self.current_poi]
        ):
            reward += r_cfg["night_penalty"]
            # done = True
            info["reason"] = "night_without_accommodation"

        if self.steps >= self.max_steps:
            done = True
            info.setdefault("reason", "max_steps")

        info["feasible_mask_next"] = self._feasible_actions_mask()
        return self._get_state(), reward, done, info
