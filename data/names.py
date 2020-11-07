import csv
import os


curr_dir = os.path.dirname(__file__)
names_dir = os.path.join(curr_dir, "names")


class NameList:

    def __init__(self, year=2018):
        filename = "yob" + str(year) + ".txt"
        filepath = os.path.join(names_dir, filename)
        
        female_names = {}
        male_names = {}

        with open(filepath) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                [name, gender, count] = row
                if gender == "F":
                    female_names[name] = int(count)
                elif gender == "M":
                    male_names[name] = int(count)

        self.female_names = female_names
        self.male_names = male_names


    def get_gender_of(self, name):
        ratio = 10
        female_count = self.female_names.get(name, 0)
        male_count = self.male_names.get(name, 0)

        if female_count > male_count * ratio:
            return "F"
        elif male_count > female_count * ratio:
            return "M"
        else:
            return "U"