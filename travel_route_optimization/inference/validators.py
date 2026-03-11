def check_opening_hours(route, opening_mask):
    violations = []
    for step in route:
        poi = step.poi_idx
        day = step.day
        start = int(step.arrival_minute)
        end =   int(step.departure_minute)

        for minute in range(start, end + 1):
            if opening_mask[poi, day, minute] != 1:
                violations.append(
                    f"{step.poi_name} closed at day {day}, minute {minute}"
                )
                break

    return len(violations) == 0, violations


def check_accommodation_per_day(route):
    violations = []
    acc_per_day = {}

    for step in route:
        if step.is_accommodation:
            acc_per_day.setdefault(step.day, 0)
            acc_per_day[step.day] += 1

    for day, count in acc_per_day.items():
        if count > 1:
            violations.append(f"Day {day} has {count} accommodations")

    return len(violations) == 0, violations


def check_lunch_per_day(route, lunch_start=660, lunch_end=900):
    violations = []
    lunch_per_day = {}

    for step in route:
        if step.category == "restauration":
            if lunch_start <= step.arrival_minute <= lunch_end:
                lunch_per_day.setdefault(step.day, 0)
                lunch_per_day[step.day] += 1

    for day, count in lunch_per_day.items():
        if count > 1:
            violations.append(f"Day {day} has {count} lunch restaurants")

    return len(violations) == 0, violations


def check_no_revisits(route):
    seen = set()
    violations = []

    for step in route:
        if step.poi_idx in seen:
            violations.append(f"POI revisited: {step.poi_name}")
        else:
            seen.add(step.poi_idx)

    return len(violations) == 0, violations


def check_day_transitions(route):
    violations = []
    prev_day = route[0].day if route else None

    for step in route[1:]:
        if step.day < prev_day:
            violations.append(
                f"Day goes backward: {prev_day} → {step.day} at {step.poi_name}"
            )
        prev_day = step.day

    return len(violations) == 0, violations


def check_travel_time_consistency(route, travel_time_matrix):
    violations = []

    for prev, curr in zip(route[:-1], route[1:]):
        expected = travel_time_matrix[prev.poi_idx, curr.poi_idx]
        actual = curr.arrival_minute - prev.departure_minute

        if abs(actual - expected) > 2:  # allow small rounding error
            violations.append(
                f"Travel mismatch {prev.poi_name} → {curr.poi_name}: "
                f"expected {expected:.1f}, got {actual:.1f}"
            )

    return len(violations) == 0, violations


def validate_route(route, opening_mask, travel_time_matrix):
    results = {}

    ok, issues = check_opening_hours(route, opening_mask)
    results["opening_hours"] = (ok, issues)

    ok, issues = check_accommodation_per_day(route)
    results["accommodation"] = (ok, issues)

    ok, issues = check_lunch_per_day(route)
    results["lunch"] = (ok, issues)

    ok, issues = check_no_revisits(route)
    results["no_revisits"] = (ok, issues)

    ok, issues = check_day_transitions(route)
    results["day_transitions"] = (ok, issues)

    ok, issues = check_travel_time_consistency(route, travel_time_matrix)
    results["travel_time"] = (ok, issues)

    return results
