import numpy as np

class BoxAction:
    """ Encodes a box movement action."""
    def __init__(self, pos_from, pos_to, zone_from, zone_to):
        self.pos_from = pos_from
        self.pos_to = pos_to
        self.zone_from = zone_from
        self.zone_to = zone_to

    def is_null(self):
        return np.array_equal(self.pos_from, self.pos_to) and self.zone_from == self.zone_to
    
    def __str__(self):
        return f"({self.pos_from} -> {self.pos_to}, {self.zone_from} -> {self.zone_to})"

    def __eq__(self, other):
        return np.array_equal(self.pos_from, other.pos_from) \
                and np.array_equal(self.pos_to, other.pos_to) \
                and self.zone_from == other.zone_from \
                and self.zone_to == other.zone_to