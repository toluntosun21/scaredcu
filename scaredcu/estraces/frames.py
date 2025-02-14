import cupy as _cp


def is_array_equivalent_to_a_slice(array):
    return array is not None and _array_has_enough_items(array) and _items_are_in_ascending_order(array) and _items_are_equally_spaced(array)


def _items_are_in_ascending_order(array):
    return _cp.any(_cp.diff(array) > 0)


def _array_has_enough_items(array):
    return len(array) >= 3


def _items_are_equally_spaced(array):
    return not _cp.any(_cp.diff(array, 2))


def build_equivalent_slice(array):
    if len(array) == 0:
        return slice(0, 0, 1)
    start = _cp.min(array)
    stop = _cp.max(array) + 1
    step = array[1] - array[0] if len(array) > 1 else 1
    return slice(start.get(), stop.get(), step.get())
