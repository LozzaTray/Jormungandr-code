import os


curr_dir = os.path.dirname(__file__)
misc = os.path.join(curr_dir, "misc")


def get_misc_path(filename):
    return os.path.join(misc, filename)