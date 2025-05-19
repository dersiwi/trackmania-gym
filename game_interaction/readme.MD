# Table of contents
- [Setup and Communication](#setup-and-communication)
  - [GameInstanceManager](#gameinstancemanager)
  - [Create Python Link](#create-python-link)
    - [Where to get this information?](#where-to-get-this-information)
- [Communication between game (server) and python (client)](#communication-between-game-server-and-python-client)
  - [Sending inputs to control car](#sending-inputs-to-control-car)
  - [Receiving Frames](#receiving-frames)
  - [Getting Simulation state](#getting-simulation-state)

# Setup and Communication

Downloaded TMNF, the modloader and installed TMInterface as wanted and described by donadigo [here](https://donadigo.com/tminterface/installation), 
more specifically; since this builds upon the [linesight repository](https://github.com/Linesight-RL/linesight) it might make more sense to follow it's [installation guide](https://linesight-rl.github.io/linesight/build/html/installation.html).


### GameInstanceManager

For now, this class is responsible for launching and closing the game via PowerShell commands. (It launches the TMLoader).

### Create Python Link
Now how to run python scripts using this setup?
From what i undestand TMInterface has to launch a TCP-Socket, to which a python instance can connect. 
The [file](https://gist.github.com/Agade09/f5b57048895b0a3a64fb0b1ae09cbd7f) `plugin/python_link.as`  has to be placed into the plugin-folder in the TMInterface folder; `C:\Users\[username]\OneDrive\Dokumente\TMInterface\Plugins`.
I modified the file such that the port is 8775 and not 0; i had to set PORT=8775 in all mentions and remove `PORT = uint16(GetVariableDouble("custom_port"));`, as this did not work for some reason. I also established a connection with a python script, however it timed out every time. 

[IMPORTANT: When launching the linesight respository, it uses `custom_port` inside the angelscript to set the port via the python script. This may be important if multiple instances of the game are running and threrefore need different ports.]

#### Where to get this information?

I went into the discord and it was a pinned answer in the programming channel; this was the exact message:

```
This is a gist with a few files from our private codebase.
https://gist.github.com/Agade09/f5b57048895b0a3a64fb0b1ae09cbd7f
With this, we reproduce a single-thread TMI1. 4.3-like API in Python.
The file python_link.as must be placed in the TMInterface folder.
The file example_script.py is a minimal functioning example showing how to open a connection.
```

## Communication between game (server) and python (client)

They communicate via binary protocol. The first message of the server is always the message type, the second is the payload of the message. 

Theoretically, the trackmania instance is acting as a simulation, sending `SC_RUN_STEP_SYNC`-message everytime a simulation step is completed. 

An example interaction is implemented in `example.py`.


### Sending inputs to control car

The car is controlled not via commands specified in the [TMInteraction mod-documentation](https://donadigo.com/tminterface/) (here, there are commands like `gas 50000` - they also work in the console inside the game when game is launched via modloader). [DISCLAIMER] i did not further explore controlling the car using such commands, since the input states work really well: 

In linesight (i.e. in `tminterface2.py`) controlling the car is done using `set_input_state`, which takes a boolean for left, right, accelerate and brake, basically acting as if a key was pressed on the keyboard. This then sets the value of the control for the next simulation step.

### Receiving Frames

Actually quite straight-forward using tminterface2:

```py
if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

    _time = iface._read_int32()

    iface.request_frame(IMG_W, IMG_H) #<--- imgs scaled automatically by trackmania
    
    # do other stuff like send inputs or commands here

    iface._respond_to_call(msgtype)

elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC): # collect frame

    frame : np.npdarray = iface.get_frame(IMG_W, IMG_H)
    iface._respond_to_call(msgtype)
```

### Getting Simulation state

The simulation state queried from the game using `iface.get_simulation_state()` which returns a `SimStateData` object. The example-interaction queries simulation state and prints it to the console. This output looks like this: 

```
=== Simulation State Snapshot ===
    Time:                786130 ms
    Position:            [529.6427612304688, 49.354209899902344, 652.0673828125]
    Velocity:            [50.89628219604492, -0.166009321808815, -0.7100085020065308]
    Rotation (YawPitchRoll): [1.582823502445449, -0.0033017142612521374, -0.004220573944300383]
    Display Speed:       183 units

Input State:
    Accelerate:          True
    Brake:               False
    Left:                False
    Right:               False
    Steer (analog):      0
    Gas (analog):        0

Race Progress:
    Race Time:           783520 ms
    Rewind Time:         783530 ms
    Num Respawns:        0
   Checkpoint Times:    [10850, 773920, -1]
    Checkpoints Passed:  3
```