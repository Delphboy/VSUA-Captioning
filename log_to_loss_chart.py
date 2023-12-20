import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_log_file_as_array(log_file) -> List[str]:
    with open(log_file, "r") as f:
        lines = f.readlines()
    return lines


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--log_file", type=str, required=True)

    config = args.parse_args()
    log_file = config.log_file
