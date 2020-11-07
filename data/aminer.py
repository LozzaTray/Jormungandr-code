import csv
import os
from data.names import NameList
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test


curr_dir = os.path.dirname(__file__)
aminer_dir = os.path.join(curr_dir, "aminer")


class AMiner:

    def __init__(self, raw = True):
        if raw:
            self.gender_by_author_id = self._read_author_genders_raw()
        else:
            self.gender_by_author_id = self._read_genders_quick()

        M = F = 0
        for gender in self.gender_by_author_id.values():
            if gender == "M":
                M += 1
            elif gender == "F":
                F += 1

        self.M = M
        self.F = F


    def _read_author_genders_raw(self):
        name_list = NameList()
        filename = "AMiner-Author.txt"
        filepath = os.path.join(aminer_dir, filename)

        outfilename = "AMiner-Author-Gender.txt"
        outfilepath = os.path.join(aminer_dir, outfilename)
        outfile = open(outfilepath, "w")

        author_id = 0
        gender_by_author_id = {}

        with open(filepath) as fp:
            for line in fp:
                if line.startswith("#index"):
                    author_id = int(line[7:])
                elif line.startswith("#n"):
                    full_name = line[3:]
                    first_name = full_name.split(" ")[0]
                    
                    gender = "U"
                    if "." not in first_name:
                        gender = name_list.get_gender_of(first_name)


                    gender_by_author_id[author_id] = gender
                    outfile.write(str(author_id) + "," + gender + "\n")

        outfile.close()
        return gender_by_author_id


    def _read_genders_quick(self):
        filename = "AMiner-Author-Gender.txt"
        filepath = os.path.join(aminer_dir, filename)

        gender_by_author_id = {}
        with open(filepath) as fp:
            reader = csv.reader(fp, delimiter=",")
            for row in reader:
                author_id = int(row[0])
                gender = row[1]
                gender_by_author_id[author_id] = gender

        return gender_by_author_id

    def check_gender_disparity(self):
        filename = "AMiner-Coauthor.txt"
        filepath = os.path.join(aminer_dir, filename)

        m2m = 0
        m2f = 0
        f2f = 0

        print("Population sizes:\nM = {} , F = {}".format(self.M, self.F))

        with open(filepath) as fp:
            for line in fp:
                [author_a, author_b, _count] = line.split() #splits by any number of whitespace
                author_a = int(author_a[1:]) # remove leading #
                author_b = int(author_b)

                gender_a = self.gender_by_author_id.get(author_a)
                gender_b = self.gender_by_author_id.get(author_b)

                if gender_a == "M" and gender_b == "M":
                    m2m += 1
                elif gender_a == "M" or gender_b == "M":
                    m2f += 1
                else:
                    f2f += 1

        m2m_max = int(self.M * (self.M + 1) / 2)
        m2f_max = self.M * self.F
        f2f_max = int(self.F * (self.F + 1) / 2)

        print("Link proportions:\nm2m={:.3e} , m2f={:.3e} , f2f={:.3e}\n".format(m2m / m2m_max, m2f / m2f_max, f2f / f2f_max))

        _t, p1 = two_samples_mean_ll_ratio(m2m_max, m2f_max, m2m, m2f)
        _t, p2 = two_samples_mean_ll_ratio(m2m_max, f2f_max, m2m, f2f)
        _t, p3 = two_samples_mean_ll_ratio(m2f_max, f2f_max, m2f, f2f)

        print("p-values for (1: m2m v m2f ; 2: m2m v f2f ; 3: m2f v f2f):")
        print("p1 = {:.3e} , p2 = {:.3e} , p3 = {:.3e}".format(float(p1), float(p2), float(p3)))