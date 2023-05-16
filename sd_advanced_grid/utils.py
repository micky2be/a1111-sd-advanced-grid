# Python
import re

# Lib
import numpy

# ################################# Constants ################################ #

FALSY = ["false", "no", "0"]
TRUTHY = ["true", "yes", "1"]

# ############################# Utility Functions ############################ #

def clean_name(name: str):
    return str(name).lower().strip().replace(" ", "_").replace("[", "").replace("]", "")

def get_closest_from_list(name: str, choices: list[str]) -> str:
    if name in choices:
        return name

    found = sorted([item for item in choices if name in item], key=len)
    if found:
        return found[0]
    return ""


re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

re_range_count = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\[(\d+)\s*\])?\s*")
re_range_count_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\[(\d+(?:.\d*)?)\s*\])?\s*")

def parse_range_int(value_list: list[str]) -> list[int]:
    parsed_list = []

    for val in value_list:
        match = re_range.fullmatch(val)
        count = re_range_count.fullmatch(val)
        if match is not None:
            start = int(match.group(1))
            end   = int(match.group(2))+1
            step  = int(match.group(3)) if match.group(3) is not None else 1

            parsed_list += list(range(start, end, step))
        elif count is not None:
            start = int(count.group(1))
            end   = int(count.group(2))
            num   = int(count.group(3)) if count.group(3) is not None else 1
            parsed_list += [int(x) for x in numpy.linspace(start=start, stop=end, num=num).tolist()]
        else:
            parsed_list.append(int(val))

    return parsed_list

def parse_range_float(value_list: list[str]) -> list[float]:
    parsed_list = []

    for val in value_list:
        match = re_range_float.fullmatch(val)
        count = re_range_count_float.fullmatch(val)
        if match is not None:
            start = float(match.group(1))
            end   = float(match.group(2))
            step  = float(match.group(3)) if match.group(3) is not None else 1
            parsed_list += numpy.arange(start, end + step, step).tolist()
        elif count is not None:
            start = float(count.group(1))
            end   = float(count.group(2))
            num   = int(count.group(3)) if count.group(3) is not None else 1
            parsed_list += numpy.linspace(start=start, stop=end, num=num).tolist()
        else:
            parsed_list += [round(float(val), 2)]

    return parsed_list
