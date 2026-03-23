import time
import threading

class TrafficController:
    def __init__(self):
        # 4 lanes, initial states
        self.lanes = {
            1: {"state": "RED", "density": 0, "emergency": False, "timer": 0},
            2: {"state": "RED", "density": 0, "emergency": False, "timer": 0},
            3: {"state": "RED", "density": 0, "emergency": False, "timer": 0},
            4: {"state": "RED", "density": 0, "emergency": False, "timer": 0},
        }
        self.current_green_lane = 1
        self.lanes[self.current_green_lane]["state"] = "GREEN"
        
        self.lock = threading.Lock()
        self.running = False
        
    def start(self):
        self.running = True
        threading.Thread(target=self._control_loop, daemon=True).start()
        
    def stop(self):
        self.running = False
        
    def update_density(self, lane, density, emergency):
        with self.lock:
            self.lanes[lane]["density"] = density
            self.lanes[lane]["emergency"] = emergency

    def _calculate_green_time(self, density):
        if density < 5:
            return 15
        elif density < 15:
            return 30
        else:
            return 45
            
    def _control_loop(self):
        while self.running:
            emergency_lane = None
            
            with self.lock:
                for lane, data in self.lanes.items():
                    if data["emergency"]:
                        emergency_lane = lane
                        break
                
                if emergency_lane and self.current_green_lane != emergency_lane:
                    self._switch_light(emergency_lane)
                    
            if emergency_lane:
                time.sleep(2) # Hold for emergency
                continue

            # Normal logic
            with self.lock:
                sorted_lanes = sorted(self.lanes.keys(), key=lambda l: self.lanes[l]["density"], reverse=True)
                target_lane = sorted_lanes[0]
                
                # Prevent starvation / handle empty lanes by falling back to round-robin
                if self.lanes[target_lane]["density"] == 0:
                    target_lane = (self.current_green_lane % 4) + 1
                    green_time = 10
                else:
                    green_time = self._calculate_green_time(self.lanes[target_lane]["density"])
                
                if self.current_green_lane != target_lane:
                    self._switch_light(target_lane)
                    
            # Wait for green_time, but break if emergency arrives
            elapsed = 0
            while elapsed < green_time and self.running:
                with self.lock:
                    if any(self.lanes[l]["emergency"] for l in self.lanes):
                        break
                time.sleep(1)
                elapsed += 1

    def _switch_light(self, target_lane):
        # Yellow phase
        if self.current_green_lane:
            self.lanes[self.current_green_lane]["state"] = "YELLOW"
            
        # We assume holding the lock so we don't sleep inside a lock typically, 
        # but for this demo, we'll quickly change. A real system uses non-blocking async logic.
        # So we release lock, sleep, re-acquire.
        self.lock.release()
        time.sleep(3) # Yellow light duration
        self.lock.acquire()
        
        # Turn old to RED, new to GREEN
        for lane in self.lanes:
            if lane == target_lane:
                self.lanes[lane]["state"] = "GREEN"
            else:
                self.lanes[lane]["state"] = "RED"
                
        self.current_green_lane = target_lane
        
    def get_status(self):
        with self.lock:
            return {l: self.lanes[l]["state"] for l in self.lanes}
            
    def simulate_emergency(self, lane):
        with self.lock:
            self.lanes[lane]["emergency"] = True
