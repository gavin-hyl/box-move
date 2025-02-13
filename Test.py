from BoxMoveEnv import BoxMoveEnv
import Box
from Dense import dense_state

def box_test():
    b = Box.make([0, 0, 0], [1, 2, 3], 0)
    print(b)
    print("pos:", Box.pos(b))
    print("size:", Box.size(b))
    print("zone:", Box.zone(b))
    print("bottom_face:", Box.bottom_face(b))
    print("top_face:", Box.top_face(b))

def bme_test():
    bme = BoxMoveEnv(n_boxes=2)
    bme.reset()

    print(bme.state)
    print(dense_state(bme.state))
    for a in bme.actions():
        print(a)

if __name__ == "__main__":
    box_test()
    # bme_test()