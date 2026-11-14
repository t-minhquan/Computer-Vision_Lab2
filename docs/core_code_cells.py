# ==================== CELL 3 ====================
import heapq
import numpy as np


def _b_integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} phải là số nguyên >= {minimum}")
    return int(value)


def _b_inputs(img, seeds):
    image = np.asarray(img, dtype=np.float64)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("img phải là ảnh xám 2D hữu hạn, không rỗng")
    points = np.asarray(seeds)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0
            or not np.issubdtype(points.dtype, np.integer)
            or np.any(points < 0) or np.any(points >= image.shape)):
        raise ValueError("seeds phải là mảng tọa độ nguyên (N, 2) nằm trong ảnh")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("seeds không được trùng vị trí")
    return image, points.astype(np.intp, copy=True)


def _b_neighbors(row, col, shape, connectivity=4):
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in offsets:
        r, c = row + dr, col + dc
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            yield r, c


def _b_srg_engine(img, seeds, connectivity=4, stale="lazy", tie="insertion", on_step=None):
    image, seeds = _b_inputs(img, seeds)
    if connectivity not in (4, 8):
        raise ValueError("connectivity phải là 4 hoặc 8")
    if stale not in ("lazy", "frozen"):
        raise ValueError("stale phải là lazy hoặc frozen")
    if tie not in ("insertion", "raster", "random"):
        raise ValueError("tie phải là insertion, raster hoặc random")
    labels = np.full(image.shape, -1, dtype=np.int32)
    boundary = np.zeros(image.shape, dtype=bool)
    for region, (row, col) in enumerate(seeds):
        labels[row, col] = region
    totals = image[seeds[:, 0], seeds[:, 1]].copy()
    seed_values = totals.copy()
    counts = np.ones(len(seeds), dtype=np.int64)
    versions = np.zeros(image.shape, dtype=np.int64)
    insertion = np.full(image.shape, -1, dtype=np.int64)
    serial = 0
    heap = []
    if tie == "random":
        pixel_priority = np.random.random(image.shape)
        region_priority = np.random.random(len(seeds))

    def best_region(row, col):
        adjacent = sorted({int(labels[r, c]) for r, c in
                           _b_neighbors(row, col, image.shape, connectivity) if labels[r, c] >= 0})
        if not adjacent:
            raise RuntimeError("Pixel frontier không kề vùng nào")
        def key(region):
            reference = totals[region] / counts[region] if stale == "lazy" else seed_values[region]
            delta = abs(float(image[row, col] - reference))
            if tie == "raster":
                priority = tuple(seeds[region])
            elif tie == "random":
                priority = (float(region_priority[region]),)
            else:
                priority = (region,)
            return (delta, *priority, region)
        winner = min(adjacent, key=key)
        return key(winner)[0], winner, len(adjacent) > 1

    def enqueue(row, col):
        nonlocal serial
        if labels[row, col] >= 0:
            return
        if insertion[row, col] < 0:
            insertion[row, col] = serial
            serial += 1
        delta, region, _ = best_region(row, col)
        versions[row, col] += 1
        if tie == "raster":
            priority = (row, col)
        elif tie == "random":
            priority = (float(pixel_priority[row, col]), int(insertion[row, col]))
        else:
            priority = (int(insertion[row, col]), 0)
        heapq.heappush(heap, (delta, *priority, row, col, int(versions[row, col]), region))

    for row, col in seeds:
        for r, c in _b_neighbors(row, col, image.shape, connectivity):
            enqueue(r, c)
    assigned = len(seeds)
    if on_step is not None:
        on_step(labels, assigned)
    while heap:
        saved_delta, _, _, row, col, version, saved_region = heapq.heappop(heap)
        if labels[row, col] >= 0 or version != versions[row, col]:
            continue
        delta, region, is_boundary = best_region(row, col)
        if delta != saved_delta or region != saved_region:
            enqueue(row, col)
            continue
        labels[row, col] = region
        boundary[row, col] = is_boundary
        totals[region] += image[row, col]
        counts[region] += 1
        assigned += 1
        if on_step is not None:
            on_step(labels, assigned)
        for r, c in _b_neighbors(row, col, image.shape, connectivity):
            enqueue(r, c)
    return labels, boundary


# ==================== CELL 6 ====================
def region_growing_classic(img, seeds, T, order="stack"):
    """Phát triển từng vùng, lân cận 4; ngưỡng so với trung bình cập nhật.

    Khởi tạo tất cả seed trước, xử lý vùng theo thứ tự seeds. Frontier
    chứa pixel đã nhận: duyệt lân cận, xét ngưỡng và cập nhật trung bình
    ngay trước khi đưa pixel lân cận được nhận vào stack/queue (PDF trang 6).
    """
    image, seeds = _b_inputs(img, seeds)
    if not np.isfinite(T) or T < 0:
        raise ValueError("T phải không âm và hữu hạn")
    if order not in ("stack", "queue"):
        raise ValueError("order phải là stack hoặc queue")
    labels = np.full(image.shape, -1, dtype=np.int32)
    for region, (row, col) in enumerate(seeds):
        labels[row, col] = region
    for region, (row, col) in enumerate(seeds):
        total, count = float(image[row, col]), 1
        frontier = [(row, col)]
        head = 0
        while frontier if order == "stack" else head < len(frontier):
            if order == "stack":
                r, c = frontier.pop()
            else:
                r, c = frontier[head]
                head += 1
            for nr, nc in _b_neighbors(r, c, image.shape):
                if labels[nr, nc] < 0 and abs(float(image[nr, nc]) - total / count) <= T:
                    labels[nr, nc] = region
                    total += image[nr, nc]
                    count += 1
                    frontier.append((nr, nc))
    return labels


# ==================== CELL 9 ====================
def srg(img, seeds, connectivity=4, stale="lazy", tie="insertion"):
    """Trả (labels int32, boundary_mask bool), mọi pixel cuối cùng có nhãn.

    lazy: tính lại delta khi pop theo trung bình vùng hiện tại, requeue nếu
    thay đổi. frozen: so với cường độ seed. Heap lazy không quét lại toàn
    bộ frontier khi trung bình đổi; ưu tiên là các khóa hiện có trong heap.
    boundary_mask ghi pixel kề nhiều vùng tại thời điểm gán, không gồm seed.
    random dùng np.random.seed để tái lập; ưu tiên ngẫu nhiên cố định trong lần chạy.
    """
    return _b_srg_engine(img, seeds, connectivity, stale, tie)


# ==================== CELL 11 ====================
def srg_step_frames(img, seeds, n_frames=8, **kwargs):
    """Chụp tại ceil(k * số_pixel / n_frames), k=1..n_frames; frame cuối hoàn tất.

    n_frames=1 chỉ chụp kết quả cuối. Nhiều mốc trùng nhau trên ảnh nhỏ
    hoặc mốc đã đạt ngay khi gán seeds vẫn tạo các bản sao độc lập.
    kwargs hỗ trợ connectivity, stale và tie.
    """
    image, seeds = _b_inputs(img, seeds)
    n_frames = _b_integer(n_frames, "n_frames", 1)
    targets = np.ceil(np.arange(1, n_frames + 1) * image.size / n_frames).astype(int)
    frames = []
    def capture(labels, assigned):
        while len(frames) < n_frames and assigned >= targets[len(frames)]:
            frames.append(labels.copy())
    _b_srg_engine(image, seeds, on_step=capture, **kwargs)
    return frames


# ==================== CELL 13 ====================
def seed_sensitivity(img, seeds, segment_fn, n_trials=20, jitter=3):
    """Tần suất đổi vùng theo định danh seed, so với phân vùng gốc.

    segment_fn(image, seeds) trả labels hoặc (labels, boundary_mask).
    Nhãn tại các seed phải khác nhau; pixel chưa gán giữ mã -1. Dịch seed
    trong cửa sổ nguyên ±jitter, giới hạn trong ảnh, không trùng; tránh các
    vị trí seed gốc còn lại để luôn có lựa chọn dự phòng. Dùng np.random.seed
    để tái lập. Mỗi lần gọi segment_fn dùng cùng trạng thái NumPy RNG
    toàn cục rồi khôi phục RNG bên ngoài, tách khỏi randomness dịch seeds.
    Callable dùng RNG riêng phải tự được cấu hình tái lập.
    """
    image, seeds = _b_inputs(img, seeds)
    n_trials = _b_integer(n_trials, "n_trials", 1)
    jitter = _b_integer(jitter, "jitter")
    if not callable(segment_fn):
        raise ValueError("segment_fn phải là callable")
    segmentation_rng_state = np.random.get_state()
    def segment(points):
        outer_rng_state = np.random.get_state()
        try:
            np.random.set_state(segmentation_rng_state)
            output = segment_fn(image.copy(), points.copy())
        finally:
            np.random.set_state(outer_rng_state)
        labels = np.asarray(output[0] if isinstance(output, tuple) else output)
        if (labels.shape != image.shape or not np.issubdtype(labels.dtype, np.integer)
                or np.any(labels < -1)):
            raise ValueError("segment_fn phải trả ảnh nhãn nguyên cùng shape; nhãn >= -1")
        ids = labels[points[:, 0], points[:, 1]]
        if np.any(ids < 0) or len(np.unique(ids)) != len(points):
            raise ValueError("mỗi seed phải thuộc một vùng khác nhau")
        normalized = np.full(image.shape, -1, dtype=np.int32)
        for region, label in enumerate(ids):
            normalized[labels == label] = region
        if np.any((labels >= 0) & (normalized < 0)):
            raise ValueError("vùng có nhãn phải tương ứng với một seed")
        return normalized
    baseline = segment(seeds)
    changed = np.zeros(image.shape, dtype=np.float64)
    for _ in range(n_trials):
        moved = seeds.copy()
        occupied = {tuple(point) for point in seeds}
        for index, (row, col) in enumerate(seeds):
            occupied.remove(tuple(seeds[index]))
            candidates = [(r, c)
                          for r in range(max(0, row - jitter), min(image.shape[0], row + jitter + 1))
                          for c in range(max(0, col - jitter), min(image.shape[1], col + jitter + 1))
                          if (r, c) not in occupied]
            choice = candidates[np.random.randint(len(candidates))] if candidates else (row, col)
            moved[index] = choice
            occupied.add(choice)
        changed += segment(moved) != baseline
    return changed / n_trials


