
# Process Wrapper

The process wrapper handles communication between python client and game-instance. In order to do this seperately from the main program, it is launched in it's own proocessed; hence called process wrapper.
The communication between the process-wrapper and another part of the process (in another process) is done via a command- and response queue.

## Modi
The Process Wrapper basically has two modi. The first one; which is 'regular' basically just runs through the main-communication loop and handles each command as soon as they are deteceted in the command-queue. Directly sends a response.

The second mode is the 'waitforstep'-Mode, which can be activated by sending `TMIProcessWrapper.IPCCommands.waitforstep()`. In this mode, the wrapper exepcts `IPCCommands.step` - commadnds. It waits with the execution of the main communication for as long as no step command has been sent. This has been implemented for reinforcement learning environment purposes.

## IPC-Protocol
This is the 'protocol' which defines communication between process wrapper and other processes. Available commands can be created using `TMIProcessWrapper.IPCCommands`. The process wrapper alaways sends an answer. The answer can be identified, by comparing the command-id of the command and the response. They should be equal.