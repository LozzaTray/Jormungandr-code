from typing import List


class Player:
    
    
    def __init__(self, csv_fields: List[str]):
        self.team = int(csv_fields[0])
        self.attendances = csv_fields.count("TRUE")
        self.absences = csv_fields.count("FALSE")


    def isFirstTeam(self):
        return self.team == 1

    def isSecondTeam(self):
        return self.team == 2