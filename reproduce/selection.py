"""Select the best strictly feasible candidate, retaining failed instances."""


def select_candidate(metrics):
    valid = [i for i, row in enumerate(metrics) if row['strict_valid']]
    return max(valid, key=lambda i: (metrics[i]['strip_utilization'], -i)) if valid else None
