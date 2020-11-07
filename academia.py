from data.names import NameList
from data.aminer import AMiner

def run():
    print("Reading in data...")
    aminer = AMiner(False)
    print("Computing graph...")
    aminer.check_gender_disparity()


if __name__ == "__main__":
    run()