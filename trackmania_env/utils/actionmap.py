
ACTION_MAP = [
        # (left, right, accelerate, brake)
        # 0 Forward
        (False,False,True,False), 
        # 1 Forward left
        (True,False,True,False),
        # 2 Forward right
        (False,True,True,False),
        # 3 Nothing
        (False,False,False,False),
        # 4 Nothing left
        (True,False,False,False),
        # 5 Nothing right
        (False,True,False,False),
        # 6 Brake
        (False,False,False,True),
        # 7 Brake left
        (True,False,False,True),
        # 8 Brake right
        (False,True,False,True),
        # 9 Brake and accelerate
        (False,False,True,True),
        # 10 Brake and accelerate left
        (True,False,True,True),
        # 11 Brake and accelerate right
        (False,True,True,True),
        ]

REVERSE_ACTION_MAP: dict[tuple[bool, bool, bool, bool], int] = {
    action: i for i, action in enumerate(ACTION_MAP)
}

