from game_interaction.tminterface2 import MessageType, TMInterface
import numpy as np
import time

class TMIProcessWrapper:

    """
    TMIProcessWrapper encompasses the interaction between python (client) and trackmania instance (server) as a seperately executable process. 
    After initialization, the syncloop method can be started and runs the communication to not cause timeouts or blockings in other processes.

    Methods
    --------
    
    While this is running, the methodds
        - request_image
        - get_image
        - get_simstate
        - get_image_and_simstate
        - act

    can be used to interact with the (this class and) trackmania instance.

    Inner Workings
    --------------
    TODO
    """


    def __init__(self, tminterface : TMInterface,  img_width : int, img_height : int, img_req_frequency : int, img_store_capacity : int = 100):

        self.iface : TMInterface = tminterface


        self.img_req_frequency : int = img_req_frequency


        self.img_width = img_width
        self.img_height = img_height

        
        self.img_store_capacity : int = img_store_capacity

        self.simulation_steps : list[int] = [0 for i in range(img_store_capacity)]
        self.requested_sim_states : list[any] = [None for i in range(img_store_capacity)]

        self._req_img : bool = False
        self.__continuous_image_request : bool = False
        self._req_in_progress : bool = False
        """If True, request was sent to the tm-server but no image received yet."""
        self.requested_images : list[np.ndarray] = [None for i in range(img_store_capacity)]
        self.img_idx = 0

        self.sim_step_count = 0

        self._send_action = False
        self.action : tuple[bool, bool, bool, bool] = None
    

    def request_image(self, continuously : bool = False) -> int:
        """Request an image with the specified image and width (specified in class initialization)
        
        If continuously is True, request_image does not have to be called again and again, but rather always requests images."""
        self.__continuous_image_request = continuously
        self._req_img = True
        return self.img_idx

    def get_image(self, idx : int) -> np.ndarray:
        """Returns the requested image according to given index (idx).
        When calling request_imgage"""
        return self.requested_images[idx]
    
    def get_imgage_blocking(self, idx : int) -> np.ndarray:
        while self.requested_images[idx] is None:
            time.sleep(0.00001)
        return self.requested_images[idx]
    
    def get_simstate(self, idx : int) -> any:
        return self.requested_sim_states[idx]
    
    def get_image_and_simstate(self, idx) -> tuple[np.ndarray, any, int]:
        return self.requested_images[idx], self.requested_sim_states[idx], self.simulation_steps[idx]
        

    def __receive_frame(self):
        print("Reiving frame")
        frame = self.iface.get_frame(self.img_width, self.img_height)
        self.requested_images[self.img_idx] = frame

        assert self.simulation_steps[self.img_idx] == self.sim_step_count, f"This sould still be the same stepcount as when the image was requested, \
            but stepcount of image-receive was {self.sim_step_count} and stepcount of image-request was {self.simulation_steps[self.img_idx]}"
        
        self._req_in_progress = False

        #if self.__continuous_image_request is True, self._req_img just stays True, if its false, its reset to false and self.request_image() has to be called again.
        self._req_img = self.__continuous_image_request

        self.img_idx = (self.img_idx + 1) % self.img_store_capacity


    def __request_frame(self):
        print("Requesting frame.")
        self.iface.request_frame(self.img_width, self.img_height)
        self._req_in_progress = True

        ssD = self.iface.get_simulation_state()
        self.requested_sim_states[self.img_idx] = ssD

        self.simulation_steps[self.img_idx] = self.sim_step_count


    def act(self, action : tuple[bool, bool, bool, bool]) -> int:
        self.action = action
        self._send_action = True
        self._anticipated_simulation_step_of_execution = self.simulation_steps + 1
        return self._anticipated_simulation_step_of_execution
    
    def __send_action(self) -> None:
        left, right, acc, brake = self.action
        if not self._anticipated_simulation_step_of_execution == self.sim_step_count:
            print(f"[WARNING] : Anticipated to execute action on simulation step {self._anticipated_simulation_step_of_execution}, but actual simulation step was {self.sim_step_count}")
        self.iface.set_input_state(left, right, acc, brake)
        self._send_action = False

    def syncloop(self):

        while True:
            print(self.sim_step_count)
            msgtype = self.iface._read_int32()
            
            # ============================================= READ INCOMING MESSAGES
            
            
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

                _time = self.iface._read_int32() # _time in this case is the total simulation time (i think)
                self.sim_step_count += 1

                # ============================ BEGIN ON RUN STEP ============================

                if self._req_img and not self._req_in_progress and self.sim_step_count % self.img_req_frequency == 0:
                    self.__request_frame()

                if self._send_action:
                    self.__send_action()
                    


                """if stepcount % self.img_req_frequency == 0:
                if not inputset is None and inputcounter < len(inputset) and stepcount % INPUT_SET_FREQUENCY == 0:
                    # server should not reply to this command.
                    (left, right, acc, brake) = inputset[inputcounter]
                    inputcounter += 1
                    if inputcounter >= len(inputset) and REPEAT:
                        inputcounter = 0
                """                    

                # ============================ END ON RUN STEP ============================
                self.iface._respond_to_call(msgtype)


            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):

                current = self.iface._read_int32()
                target = self.iface._read_int32()

                # ============================ BEGIN ON CP COUNT ============================

                # ============================ END ON CP COUNT ============================
                self.iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):

                self.iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                self.__receive_frame()
                self.iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.C_SHUTDOWN):

                self.iface.close()

            elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                print("--------------------On connect event.!--------------------------")
                self.iface.on_connect_event()
                self.iface._respond_to_call(msgtype)
            else:
                self.iface._respond_to_call(msgtype)
        

        


    