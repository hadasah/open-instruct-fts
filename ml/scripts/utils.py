import collections.abc
import os
import time

from copy import deepcopy

# def recursive_dict_update(original_dict, update_dict):
#     """
#     Recursively updates a dictionary with values from another dictionary.
#     Handles nested dictionaries by merging them.
#     """
#     for key, value in update_dict.items():
#         if key in original_dict and isinstance(original_dict[key], dict) and isinstance(value, dict):
#             # If both values are dictionaries, recurse
#             original_dict[key] = recursive_dict_update(original_dict[key], value)
#         else:
#             # Otherwise, update or add the key-value pair
#             original_dict[key] = value
#     return original_dict


def seq_dict_update(ds):
    """
    Recursively update a dict with another dict.
    This is a deep update, meaning that if a key in the first dict
    has a dict as its value, and the second dict has a key with
    the same name, the value in the first dict will be updated
    with the value from the second dict.
    Keys in the second dict are the ones iterated over. 
    If the value in the second dict is not a dict, it will
    overwrite the value in the first dict.
    """
    if len(ds) == 0:
        return {}
    if len(ds) == 1:
        return deepcopy(ds[0])
    d = deepcopy(ds[0])
    for u in ds[1:]:
        if not u:
            continue
        if not d:
            d = deepcopy(u)
        else:
            u = deepcopy(u)
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = seq_dict_update([d.get(k, {}), v])
                else:
                    d[k] = v
    return d

def has_file_been_modified_recently(filepath, recent_threshold_seconds=3600):
    """
    Checks if a file has been modified within a specified recent time threshold.

    Args:
        filepath (str): The path to the file.
        recent_threshold_seconds (int): The number of seconds defining "recently".
                                       Defaults to 3600 seconds (1 hour).

    Returns:
        bool: True if the file was modified within the threshold, False otherwise.
    """
    if not os.path.exists(filepath):
        return False  # File does not exist

    last_modified_time = os.path.getmtime(filepath)
    current_time = time.time()

    return (current_time - last_modified_time) < recent_threshold_seconds
