def build_prefix_sum(M):
    """
    Build 2D prefix sums for matrix M of size N x N.
    Returns prefix_sum P, where P[r][c] = sum of M[0..r][0..c].
    """
    N = len(M)
    P = [[0] * (N + 1) for _ in range(N + 1)]
    
    for r in range(1, N + 1):
        for c in range(1, N + 1):
            P[r][c] = (M[r-1][c-1]
                       + P[r-1][c]
                       + P[r][c-1]
                       - P[r-1][c-1])
    return P

def submatrix_sum(P, r1, c1, r2, c2):
    """
    Returns sum of M[r1..r2][c1..c2] using prefix sum array P.
    (r1, c1, r2, c2) are 0-based coordinates for M.
    """
    return (P[r2+1][c2+1]
            - P[r2+1][c1]
            - P[r1][c2+1]
            + P[r1][c1])

def find_locally_maximal_rectangles(M):
    N = len(M)
    P = build_prefix_sum(M)

    left_top_cells = []
    right_bottom_cells = []

    for r1 in range(N):
        for c1 in range(N):
            for r2 in range(r1, N):
                for c2 in range(c1, N):
                    # 1) Compute sum and ratio
                    rect_sum = submatrix_sum(P, r1, c1, r2, c2)
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    ratio_current = rect_sum / area

                    # 2) Check expansions
                    if not is_locally_maximal(M, P, r1, c1, r2, c2, ratio_current):
                        continue  # skip if expansion check fails

                    # 3) If locally maximal, store corners
                    left_top_cells.append((r1, c1))
                    right_bottom_cells.append((r2, c2))

    return left_top_cells, right_bottom_cells

def is_locally_maximal(M, P, r1, c1, r2, c2, ratio_current):
    N = len(M)
    # Up expansion (if r1 > 0)
    if r1 > 0:
        ratio_up = compute_ratio(P, r1 - 1, c1, r2, c2)
        if ratio_up >= ratio_current:
            return False
    # Down expansion (if r2 < N-1)
    if r2 < N - 1:
        ratio_down = compute_ratio(P, r1, c1, r2 + 1, c2)
        if ratio_down >= ratio_current:
            return False
    # Left expansion (if c1 > 0)
    if c1 > 0:
        ratio_left = compute_ratio(P, r1, c1 - 1, r2, c2)
        if ratio_left >= ratio_current:
            return False
    # Right expansion (if c2 < N-1)
    if c2 < N - 1:
        ratio_right = compute_ratio(P, r1, c1, r2, c2 + 1)
        if ratio_right >= ratio_current:
            return False

    return True

def compute_ratio(P, r1, c1, r2, c2):
    s = submatrix_sum(P, r1, c1, r2, c2)
    area = (r2 - r1 + 1) * (c2 - c1 + 1)
    return s / area

def find_locally_maximal_rectangles(M):
    N = len(M)
    P = build_prefix_sum(M)

    left_top_cells = []
    right_bottom_cells = []

    for r1 in range(N):
        for c1 in range(N):
            for r2 in range(r1, N):
                for c2 in range(c1, N):
                    # 1) Compute sum and ratio
                    rect_sum = submatrix_sum(P, r1, c1, r2, c2)
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    ratio_current = rect_sum / area

                    # 2) Check expansions
                    if not is_locally_maximal(M, P, r1, c1, r2, c2, ratio_current):
                        continue  # skip if expansion check fails

                    # 3) If locally maximal, store corners
                    left_top_cells.append((r1, c1))
                    right_bottom_cells.append((r2, c2))

    return left_top_cells, right_bottom_cells

def is_locally_maximal(M, P, r1, c1, r2, c2, ratio_current):
    N = len(M)
    # Up expansion (if r1 > 0)
    if r1 > 0:
        ratio_up = compute_ratio(P, r1 - 1, c1, r2, c2)
        if ratio_up >= ratio_current:
            return False
    # Down expansion (if r2 < N-1)
    if r2 < N - 1:
        ratio_down = compute_ratio(P, r1, c1, r2 + 1, c2)
        if ratio_down >= ratio_current:
            return False
    # Left expansion (if c1 > 0)
    if c1 > 0:
        ratio_left = compute_ratio(P, r1, c1 - 1, r2, c2)
        if ratio_left >= ratio_current:
            return False
    # Right expansion (if c2 < N-1)
    if c2 < N - 1:
        ratio_right = compute_ratio(P, r1, c1, r2, c2 + 1)
        if ratio_right >= ratio_current:
            return False

    return True

def select_non_overlapping_rectangles(left_top_cells, right_bottom_cells, M, P):
    """
    left_top_cells[i] = (r1, c1)
    right_bottom_cells[i] = (r2, c2)
    Sort them by descending ratio, then pick greedily.
    """
    # 1) Combine corners and compute ratio
    rectangles = []
    for (r1, c1), (r2, c2) in zip(left_top_cells, right_bottom_cells):
        s = submatrix_sum(P, r1, c1, r2, c2)
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        rect_ratio = s / area
        rectangles.append(((r1, c1, r2, c2), rect_ratio))

    # 2) Sort by ratio descending (or any other criterion)
    rectangles.sort(key=lambda x: x[1], reverse=True)

    # 3) Greedy selection
    selected = []
    for rect, ratio in rectangles:
        if all(not overlap(rect, srect[0]) for srect in selected):
            selected.append((rect, ratio))

    # Unpack into final corner lists
    final_left_top = []
    final_right_bottom = []
    for (r1, c1, r2, c2), _ in selected:
        final_left_top.append((r1, c1))
        final_right_bottom.append((r2, c2))

    return final_left_top, final_right_bottom


def overlap(rect1, rect2):
    r1, c1, r2, c2 = rect1
    r1p, c1p, r2p, c2p = rect2
    # Two rectangles do not overlap if one is entirely above/below or left/right of the other.
    # So they overlap if all these conditions are false:
    return not (r2 < r1p or r2p < r1 or c2 < c1p or c2p < c1)

def find_non_overlapping_locally_maximal_rectangles(M):
    # 1) Find all locally maximal submatrices
    left_top_cells, right_bottom_cells = find_locally_maximal_rectangles(M)

    # 2) Optionally filter them to be non-overlapping (greedy)
    P = build_prefix_sum(M)
    final_lt, final_rb = select_non_overlapping_rectangles(left_top_cells,
                                                           right_bottom_cells,
                                                           M, P)
    return final_lt, final_rb


