from data.names import NameList

def run():
    name_list = NameList()
    print(name_list.get_gender_of("Brian"))
    print(name_list.get_gender_of("Alice"))
    print(name_list.get_gender_of("kajshkjha"))


if __name__ == "__main__":
    run()