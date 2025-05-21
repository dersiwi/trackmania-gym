import gymnasium as gym
import numpy as np
from bytefield import ByteStruct, FloatField, IntegerField, BooleanField, ArrayField, ByteArrayField, StructField, StringField
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

def flatten_struct(cls, prefix=""):
    """Flatten a ByteStruct class into a dict of {flat_name: gym.spaces.Box}."""
    instance = cls()  # instantiate to access resolved fields
    flat_dict = {}
    members = [(attr,getattr(instance, attr)) for attr in dir(instance) if not callable(getattr(instance, attr)) and not attr.startswith("__")]
    for (name,field) in members:
        if not isinstance(field, (IntegerField, FloatField, BooleanField, ArrayField, StructField, ByteArrayField)):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(field, StructField):
            sub_cls = field.struct_type
            flat_dict.update(flatten_struct(sub_cls, prefix=full_name))
        elif isinstance(field, ArrayField):
            shape = field.shape if field.shape is not None else (1,)  # default fallback
            try:
                dtype = get_numpy_dtype(field.elem_field_type)
            except: continue
            flat_dict[full_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)
        elif isinstance(field, (IntegerField, FloatField, BooleanField)):
            dtype = get_numpy_dtype(type(field))
            low = 0 if dtype == np.bool_ else -np.inf
            high = 1 if dtype == np.bool_ else np.inf
            flat_dict[full_name] = gym.spaces.Box(low=low, high=high, shape=(), dtype=dtype)
        elif isinstance(field, ByteArrayField):
            flat_dict[full_name] = gym.spaces.Box(low=0, high=255, shape=(field.size,), dtype=np.uint8)

    return flat_dict

def get_numpy_dtype(field_type):
    if field_type == IntegerField:
        return np.int32
    elif field_type == FloatField:
        return np.float32
    elif field_type == BooleanField:
        return np.bool_
    else:
        return np.float32  # fallback


# Usage: Flatten all SimStateData fields into a dict of gym.spaces.Box
def write_space_dict_to_file(filename: str):
    space_dict = flatten_struct(SimStateData)

    with open(filename, 'w') as f:
        f.write("from gymnasium.spaces import Box \n")
        f.write("from numpy import uint8,int32,float32,inf\n")
        f.write("simstate_space_dict = {\n")
        for key, space in space_dict.items():
            f.write(f'    "{key}": {repr(space)}  ,\n')
        f.write("}\n")


write_space_dict_to_file("simstate_space_dict.py")
if False:
    space_dict = flatten_struct(SimStateData)
    f = open( 'simstate_space_dict.py', 'w' )
    f.write("import gym\n")
    f.write("import numpy as np\n\n")
    f.write("simstate_space_dict = {\n")
    f.write( 'dict = ' + repr(space_dict) + '\n' )
    f.close()

