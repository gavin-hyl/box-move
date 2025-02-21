from BoxMoveEnv import BoxMoveEnv
import Box

def box_test():
    print("===Box Test===")
    b = Box.make([0, 0, 0], [1, 2, 3], 0)
    print(b)
    print("pos:", Box.pos(b))
    print("size:", Box.size(b))
    print("zone:", Box.zone(b))
    print("bottom_face:", Box.bottom_face(b))
    print("top_face:", Box.top_face(b))

def bme_test():
    print("===BoxMoveEnv Test===")
    bme = BoxMoveEnv(n_boxes=20)
    bme.reset()
    print("Zone 1 top:", bme.zone1_top)
    print("Boxes:", Box.boxes_from_state(bme.state))

    bme.visualize_scene()
    # for a in bme.actions():
    #     print(a)
    # if len(bme.actions()) == 0:
    #     print("No actions found")

if __name__ == "__main__":
    box_test()
    bme_test()