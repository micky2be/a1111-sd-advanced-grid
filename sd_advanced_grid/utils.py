import os, glob, math, re

from git.repo import Repo
import numpy

######################### Constants #########################
ASSET_DIR = os.path.dirname(__file__) + "/assets" # TODO: change to actual path
VERSION = ""
FALSY = ["false", "no", "0"]
TRUTHY = ["true", "yes", "1"]


## Utilities

def get_version():
    global VERSION
    if VERSION is not None:
        return VERSION
    repo = Repo(path=os.path.dirname(__file__))
    VERSION = repo.head.commit.hexsha[:8]
    return VERSION


def clean_name(name: str):
    return str(name).lower().strip().replace(' ', '_').replace('[', '').replace(']', '')

def get_closest_from_list(name: str, list: list) -> str:
    if name in list:
        return name

    found = sorted([item for item in list if name in item], key=lambda x: len(x))
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
        m = re_range.fullmatch(val)
        mc = re_range_count.fullmatch(val)
        if m is not None:
            start = int(m.group(1))
            end   = int(m.group(2))+1
            step  = int(m.group(3)) if m.group(3) is not None else 1

            parsed_list += list(range(start, end, step))
        elif mc is not None:
            start = int(mc.group(1))
            end   = int(mc.group(2))
            num   = int(mc.group(3)) if mc.group(3) is not None else 1
            
            parsed_list += [int(x) for x in numpy.linspace(start=start, stop=end, num=num).tolist()]
        else:
            parsed_list.append(int(val))

    return parsed_list

def parse_range_float(value_list: list[str]) -> list[float]:
    parsed_list = []

    for val in value_list:
        m = re_range_float.fullmatch(val)
        mc = re_range_count_float.fullmatch(val)
        if m is not None:
            start = float(m.group(1))
            end   = float(m.group(2))
            step  = float(m.group(3)) if m.group(3) is not None else 1

            parsed_list += numpy.arange(start, end + step, step).tolist()
        elif mc is not None:
            start = float(mc.group(1))
            end   = float(mc.group(2))
            num   = int(mc.group(3)) if mc.group(3) is not None else 1
            
            parsed_list += numpy.linspace(start=start, stop=end, num=num).tolist()
        else:
            parsed_list += [round(float(val), 2)]
            
    return parsed_list
