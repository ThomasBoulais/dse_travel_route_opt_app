CATEGORY_SCORES = {
    "cultural, historical & religious events or sites": 1.0,
    "parks, garden & nature": 0.8,
    "leisure & entertainment": 0.7,
    "restauration": 0.3,
    "sportive": 0.4,
    "accomodation": 0.0,
    "transport & mobility": -0.2,
    "utilitaries": -0.2,
    "": -0.5,
}


def total_interest(route):
    total = 0.0
    for step in route:
        if hasattr(step, "interest_score"):
            total += float(step.interest_score)
        else:
            total += CATEGORY_SCORES.get(step.category, 0.0)
    return total


def category_diversity(route, max_categories=10):
    categories = {step.category for step in route}
    return len(categories) / max_categories


def num_pois(route):
    return len(route)


def total_travel_time(route):
    total = 0.0
    for step in route:
        total += float(step.travel_time)
    return total


def score_route(route):
    return (
        0.4 * total_interest(route)
        + 0.2 * category_diversity(route)
        + 0.2 * num_pois(route)
        - 0.2 * total_travel_time(route)
    )
