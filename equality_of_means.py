from data.read import read_csv
from model.player import Player
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio, students_z_test

significance_level = 95

def run():
    players: List[Player] = read_csv("training_attendance.csv", Player)
    n = m = k = l = 0

    for player in players:
        if player.isFirstTeam():
            k += player.attendances
            n += player.attendances + player.absences
        elif player.isSecondTeam():
            l += player.attendances
            m += player.attendances + player.absences

    t, p = two_samples_mean_ll_ratio(n, m, k, l, debug=True)
    z, p = students_z_test(n, m, k, l, debug=True)

    if (100*p < (100 - significance_level)):
        print("Null Hypothesis rejected at the {}% significance level".format(significance_level))
        print("Means are different")
    else:
        print("Insufficient evidence to reject null")

    

if __name__ == "__main__":
    print("Testing equality of means")
    run()