from data.read import read_csv
from model.player import Player
from typing import List
from hypothesis.test_statistics import two_samples_mean_ll_ratio


def run():
    players: List[Player] = read_csv("training_attendance.csv", Player)
    n = m = k = l = 0

    for player in players:
        if player.isFirstTeam():
            k += player.attendances
            n += player.attendances + player.absences
        else:
            l += player.attendances
            m += player.attendances + player.absences
    
    t = two_samples_mean_ll_ratio(n, m, k, l, debug=True)

    

if __name__ == "__main__":
    print("Testing equality of means")
    run()