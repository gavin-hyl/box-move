import numpy as np
from Constants import GEOM_DIM


class Box:
    def __init__(self, pos=None, size=None, zone=None, val=None):
        self.pos = pos if pos is not None else np.zeros(GEOM_DIM)
        self.size = size if size is not None else np.zeros(GEOM_DIM)
        self.zone = zone if zone is not None else -1
        self.val = val if val is not None else 0
    
    def __eq__(self, other):
        return self.pos == other.pos and self.size == other.size and self.zone == other.zone and self.val == other.val
    
    def __str__(self):
        return f"pos={self.pos}, size={self.size}, zone={self.zone}, val={self.val}"
    
    def __repr__(self):
        return str(self)

    def top_face(self):
        """ Returns a set of points representing the top face of the box. Each tile is represented by its bottom-left corner. """
        points = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                points.append(self.pos + np.array([i, j, self.size[2]]))
        points = [tuple(point) for point in points]
        return set(points)

    def bottom_face(self):
        """ Returns a set of points representing the bottom face of the box. Each tile is represented by its bottom-left corner. """
        points = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                points.append(self.pos + np.array([i, j, 0]))
        points = [tuple(point) for point in points]
        return set(points)
