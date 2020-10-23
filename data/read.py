import os
import csv


curr_dir = os.path.dirname(__file__)


def read_file(filename):
    if filename.endswith(".csv"):
        return read_csv(filename)


def read_csv(filename, transform_function=lambda x: x):
    filepath = os.path.join(curr_dir, filename)
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        rows = [transform_function(row) for row in reader]
    return rows