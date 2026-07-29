

def create_split(regular_windows, faulty_windows, num_reg, num_faulty):
    if num_reg < 0 or num_faulty < 0:
        raise ValueError('num_reg and num_faulty must be non-negative')

    if num_reg > len(regular_windows):
        raise ValueError('requested too many regular windows')

    if num_faulty > len(faulty_windows):
        raise ValueError('requested too many faulty windows')

    reg_part = regular_windows[-num_reg:] if num_reg else []
    faulty_part = faulty_windows[-num_faulty:] if num_faulty else []

    if num_reg:
        del regular_windows[-num_reg:]
    if num_faulty:
        del faulty_windows[-num_faulty:]

    return reg_part + faulty_part


def contamination_train_val_test_split(regular_windows, faulty_windows, anomaly_ratio_train, val_ratio, test_ratio):
    if not (0 <= val_ratio <= 1):
        raise ValueError('val_ratio must be in [0, 1]')

    if not (0 <= test_ratio <= 1):
        raise ValueError('test_ratio must be in [0, 1]')

    if val_ratio + test_ratio > 1:
        raise ValueError('val_ratio + test_ratio must not exceed 1')

    if not (0 <= anomaly_ratio_train <= 1):
        raise ValueError('anomaly_ratio_train must be in [0, 1]')

    n_total = len(faulty_windows) + len(regular_windows)

    if n_total == 0:
        raise ValueError('number of total windows is zero')

    original_anomaly_ratio = len(faulty_windows) / n_total

    n_test = round(n_total * test_ratio)
    n_val = round(n_total * val_ratio)
    n_train = n_total - n_test - n_val

    if n_train < 0:
        raise ValueError('rounded validation and test sizes exceed n_total')

    n_faulty_test = round(n_test * original_anomaly_ratio)
    n_faulty_val = round(n_val * original_anomaly_ratio)
    n_faulty_train = round(n_train * anomaly_ratio_train)

    test = create_split(regular_windows, faulty_windows, num_reg=n_test - n_faulty_test, num_faulty=n_faulty_test)
    val = create_split(regular_windows, faulty_windows, num_reg=n_val - n_faulty_val, num_faulty=n_faulty_val)
    train = create_contaminated_training_split(regular_windows, faulty_windows, num_reg=n_train - n_faulty_train, num_faulty=n_faulty_train)

    return train, val, test


def cyclic_sample_with_replacement(available_items, requested_items):
    if requested_items < 0:
        raise ValueError('requested_items must not be negative')

    if requested_items == 0:
        return []

    if available_items <= 0:
        raise ValueError('available_items must be positive')

    return [i % available_items for i in range(requested_items)]


def create_contaminated_training_split(regular_windows, faulty_windows, num_reg, num_faulty):
    regular_windows = list(regular_windows)
    faulty_windows = list(faulty_windows)

    if num_reg > 0 and len(regular_windows) == 0:
        raise ValueError('cannot create contaminated training split because regular_windows is empty')

    if num_faulty > 0 and len(faulty_windows) == 0:
        raise ValueError('cannot create contaminated training split because faulty_windows is empty')

    regular_indices = cyclic_sample_with_replacement(len(regular_windows), num_reg)
    faulty_indices = cyclic_sample_with_replacement(len(faulty_windows), num_faulty)

    selected_regular = [regular_windows[i] for i in regular_indices]
    selected_faulty = [faulty_windows[i] for i in faulty_indices]

    train = selected_regular + selected_faulty

    if len(train) != num_reg + num_faulty:
        raise RuntimeError('training split length mismatch')

    return train
