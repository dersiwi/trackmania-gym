# Setup and Communication

Downloaded TMNF, the modloader and installed TMInterface as wanted and described here: https://donadigo.com/tminterface/installation, 
more specifically; since this builds upon the linesight repository (https://github.com/Linesight-RL/linesight) it might make more sense to follow this installation guide: https://linesight-rl.github.io/linesight/build/html/installation.html.


### GameInstanceManager

For now, this class is responsible for launching and closing the game via PowerShell commands. (It launches the TMLoader).

### Create Python Link
Now how to run python scripts using this setup?
From what i undestand TMInterface has to launch a TCP-Socket, to which a python instance can connect. 
The file `plugin/python_link.as` (i got this file from here: https://gist.github.com/Agade09/f5b57048895b0a3a64fb0b1ae09cbd7f) has to be placed into the plugin-folder in the TMInterface folder; `C:\Users\[username]\OneDrive\Dokumente\TMInterface\Plugins`.
I modified the file such that the port is 8775 and not 0; i had to set PORT=8775 in all mentions and remove `PORT = uint16(GetVariableDouble("custom_port"));`, as this did not work for some reason. I also established a connection with a python script, however it timed out every time. [IMPORTANT: When launching the linesight respository, it uses custom_port to set the port via the python script. This may be important if multiple instances of the game are running and threrefore need different ports.]

#### Where to get this information?

I went into the discord and it was a pinned answer in the programming channel; this was the exact message:

```
This is a gist with a few files from our private codebase.
https://gist.github.com/Agade09/f5b57048895b0a3a64fb0b1ae09cbd7f
With this, we reproduce a single-thread TMI1. 4.3-like API in Python.
The file python_link.as must be placed in the TMInterface folder.
The file example_script.py is a minimal functioning example showing how to open a connection.
```

