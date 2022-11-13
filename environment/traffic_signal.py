import numpy as np
from gym import spaces


class TrafficSignal:
    def __init__(
        self,
        ts_id: str,
        yellow_time: int,
        simulation_time: float,
        delta_rs_update_time: int,
        reward_fn: str,
        sumo
    ):
        self.ts_id = ts_id
        self.yellow_time = yellow_time
        self.simulation_time = simulation_time
        self.delta_rs_update_time = delta_rs_update_time
        # reward_state_update_time
        self.rs_update_time = 0
        self.reward_fn = reward_fn
        self.sumo = sumo
        self.green_phase = None
        self.yellow_phase = None
        self.end_min_time = 0
        self.end_max_time = 0
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state]
        self.num_green_phases = len(self.all_green_phases)
        self.lanes_id = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes_id}
        self.observation_space = spaces.Box(
            low=np.zeros(len(self.lanes_id), dtype=np.float32),
            high=np.ones(len(self.lanes_id), dtype=np.float32))
        self.action_space = spaces.Discrete(self.num_green_phases)
        self.last_measure = 0
        self.dict_lane_veh = None

    def change_phase(self, new_green_phase):
        """
        :param new_green_phase:
        :return: do_action -> the real action operated; if is None, means the new_green_phase is not appropriate,
        need to choose another green_phase and operate again
        """
        # yellow_phase has not finished yet
        # yellow_phase only has duration, no minDur or maxDur

        # do_action mapping (int -> Phase)
        new_green_phase = self.all_green_phases[new_green_phase]
        do_action = new_green_phase
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is not None:
            if current_time >= self.end_max_time:
                self.yellow_phase = None
                self.update_end_time()
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                do_action = self.green_phase
            else:
                do_action = self.yellow_phase
        else:
            # if old_green_phase has finished
            if current_time >= self.end_min_time:
                if new_green_phase.state == self.green_phase.state:
                    if current_time < self.end_max_time:
                        do_action = self.green_phase
                    else:
                        # current phase has reached the max operation time, have to find another green_phase instead
                        do_action = None
                else:
                    # need to set a new plan(yellow + new_green)
                    yellow_state = ''
                    for s in range(len(new_green_phase.state)):
                        if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                            yellow_state += 'y'
                        else:
                            yellow_state += self.green_phase.state[s]
                    self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                    self.green_phase = new_green_phase
                    self.rs_update_time = current_time + self.yellow_time + self.delta_rs_update_time
                    self.update_end_time()
                    do_action = self.yellow_phase
            else:
                do_action = self.green_phase

        if do_action is None:
            return None

        # do_action mapping (Phase -> int)
        if 'y' in do_action.state:
            do_action = -1
        else:
            for i, green_phase in enumerate(self.all_green_phases):
                if do_action.state == green_phase.state:
                    do_action = i
                    break

        return do_action

    def update_end_time(self):
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is None:
            self.end_min_time = current_time + self.green_phase.minDur
            self.end_max_time = current_time + self.green_phase.maxDur
        else:
            self.end_min_time = current_time + self.yellow_time
            self.end_max_time = current_time + self.yellow_time

    def compute_reward(self, start, do_action):
        update_reward = False
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            self.rs_update_time = self.simulation_time + self.delta_rs_update_time
            update_reward = True
        if self.reward_fn == 'choose-min-waiting-time':
            return self._choose_min_waiting_time(start, update_reward, do_action)
        else:
            return None

    def _choose_min_waiting_time(self, start, update_reward, do_action):
        if start:
            self.dict_lane_veh = {}
            for lane_id in self.lanes_id:
                veh_list = self.sumo.lane.getLastStepVehicleIDs(lane_id)
                wait_veh_list = [veh_id for veh_id in veh_list if self.sumo.vehicle.getAccumulatedWaitingTime(veh_id)>0]
                self.dict_lane_veh[lane_id] = len(wait_veh_list)
            # merge wait_time by actions
            dict_action_wait_time = [self.dict_lane_veh['n_t_0'] + self.dict_lane_veh['s_t_0'],
                                     self.dict_lane_veh['n_t_1'] + self.dict_lane_veh['s_t_1'],
                                     self.dict_lane_veh['e_t_0'] + self.dict_lane_veh['w_t_0'],
                                     self.dict_lane_veh['e_t_1'] + self.dict_lane_veh['w_t_1']]
            best_action = np.argmax(dict_action_wait_time)
            if best_action == do_action:
                self.last_measure = 1
            else:
                self.last_measure = -1

        if update_reward:
            return self.last_measure
        else:
            return None

    def compute_next_state(self):
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            density = self.get_lanes_density()
            next_state = np.array(density, dtype=np.float32)
            return next_state
        else:
            return None

    def compute_state(self):
        density = self.get_lanes_density()
        state = np.array(density, dtype=np.float32)
        return state

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane_id) / (self.lanes_length[lane_id] / vehicle_size_min_gap))
                for lane_id in self.lanes_id]

    def compute_queue(self):
        total_queue = 0
        for lane_id in self.lanes_id:
            total_queue += self.sumo.lane.getLastStepHaltingNumber(lane_id)
        return total_queue
