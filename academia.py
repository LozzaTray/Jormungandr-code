from data.names import NameList
from data.aminer import AMiner

def run():
    print("Reading in names...")
    name_list = NameList()
    print("Reading in authors...")
    aminer = AMiner(name_list)


if __name__ == "__main__":
    run()