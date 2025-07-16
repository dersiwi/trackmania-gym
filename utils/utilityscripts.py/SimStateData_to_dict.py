import gymnasium as gym
import numpy as np
from bytefield import (
    FloatField, 
    IntegerField, 
    BooleanField, 
    ArrayField, 
    ByteArrayField, 
    StructField,
    ArrayFieldProxy, 
    ByteArrayFieldProxy, 
    ByteStruct)
from typing import List
from tminterface.structs import SimStateData,SimulationWheel,WheelState
banned = ["last_field","input_steer_event.time_field","internal_input_state","last"]

def is_list_of_strings(obj):
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)

def flatten_struct(cls, prefix=""):
    """Flatten a ByteStruct class into a dict of {flat_name: gym.spaces.Box}."""
    instance = cls()  # instantiate to access resolved fields
    flat_dict = {}
    members = [(attr,getattr(instance, attr)) 
                for attr in dir(instance) 
                if not callable(getattr(instance, attr)) 
                and not attr.startswith("__")]
    
    for (name,field) in members:
        full_name = f"{prefix}.{name}" if prefix else name

        if (not isinstance(field, (
            IntegerField,
            FloatField,
            BooleanField,
            ArrayField,
            StructField,
            ByteStruct,
            ByteArrayField,
            ArrayFieldProxy,
            ByteArrayFieldProxy,
            List,
            bytearray,
            int,
            float,
            bool)) 
            or (full_name in banned)) or (name in banned) or is_list_of_strings(field): 
            continue
     
        if isinstance(field,ByteStruct):
            flat_dict.update(flatten_struct(type(field), prefix=full_name))
    
        elif isinstance(field, StructField):
            sub_cls = field.struct_type 
            full_name = full_name.replace("_field","")
            flat_dict.update(flatten_struct(sub_cls, prefix=full_name))
        
        elif isinstance(field,bytearray):
            # assume that every bytearray is float type since there are no informations in a bytearray about the type
            l = list(field)
            flat_dict[full_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(l),), dtype=np.float32)
            
        elif isinstance(field,ArrayFieldProxy):
            #TODO need to do correct type handling
            if full_name == "simulation_wheels":
                pass

            if not isinstance(field.field._elem_field,(FloatField,IntegerField,BooleanField)):
                d = {}
                for i in range(len(field.to_numpy())):
                    d.update(flatten_struct(type(field[i]),full_name+f"[{i}]"))
                flat_dict[full_name] = flat_dict.update(d)    
                continue
            dtype = get_numpy_dtype(type(field.field._elem_field))
            low = 0 if dtype == np.bool_ else -np.inf
            high = 1 if dtype == np.bool_ else np.inf
            flat_dict[full_name] = gym.spaces.Box(low=low, high=high, shape=field.shape, dtype=dtype)

        elif isinstance(field,ByteArrayFieldProxy):
            l = list(field.to_bytearray())
            # assume that we are dealing with homogenous lists
            dtype = type(l[0])
            if dtype == bool:
                flat_dict[full_name] = gym.spaces.MultiBinary(n=len(l))
            else:
                flat_dict[full_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(l),), dtype=dtype)

        elif isinstance(field, ArrayField):
            shape = field.shape if field.shape is not None else (1,)  # default fallback
            try:
                dtype = get_numpy_dtype(field.elem_field_type)
            except: continue
            flat_dict[full_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)
            
        elif isinstance(field, (IntegerField, FloatField, BooleanField,bool,int,float)):
            dtype = get_numpy_dtype(type(field))
            if dtype == np.bool_ :
                flat_dict[full_name] = gym.spaces.Discrete(2)
            else :
                # Signed check
                low = 0  if np.issubdtype(dtype, np.unsignedinteger) else -np.inf
                low = -np.inf 
                high = np.inf
                flat_dict[full_name] = gym.spaces.Box(low=low, high=high, shape=(), dtype=dtype)
            
        elif isinstance(field, ByteArrayField):
            if full_name.endswith("last") :
                full_name = full_name+"field"
            flat_dict[full_name] = gym.spaces.Box(low=0, high=255, shape=(field.size,), dtype=np.uint8)
        
        elif isinstance(field,List):
            if field == []: continue
            dtype = get_numpy_dtype(type(field[0]))
            low = 0 if dtype == np.bool_ else -np.inf
            high = 1 if dtype == np.bool_ else np.inf
            flat_dict[full_name] = gym.spaces.Box(low=low, high=high, shape=(np.shape(field)), dtype=dtype)

    # tminterface by default returns a BGRA image so 4 dimensional
    if isinstance(field,SimStateData):
        flat_dict["image"] = gym.spaces.Box(low=0, high=255, shape=(100,100,4), dtype=np.uint8)       
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
        f.write("from gymnasium.spaces import Box,Discrete,Dict \n")
        f.write("from numpy import uint8,int32,int64,float32,inf\n")
        f.write("simstate_space_dict = {\n")
        for key, space in space_dict.items():
            if space is None or space.shape == [] : continue
            f.write(f'"{key}": {repr(space)},\n')
        f.write("}\n")




if __name__ == "__main__":
    write_space_dict_to_file("simstate_space_dict.py")
    space_dict = flatten_struct(SimStateData)
    f = open( 'simstate_space_dict.py', 'w' )
    f.write("import gym\n")
    f.write("import numpy as np\n\n")
    f.write("simstate_space_dict = {\n")
    f.write( 'dict = ' + repr(space_dict) + '\n' )
    f.close()

