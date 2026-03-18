from dataclasses import dataclass


@dataclass
class RouteStep:
    poi_idx: int
    poi_name: str
    day: int
    arrival_minute: int
    departure_minute: int
    category: str
    is_accommodation: bool
    travel_time: float
    visit_duration: float
    interest_score: float
