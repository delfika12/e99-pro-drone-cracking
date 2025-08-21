from DroneController import Drone

drone = Drone()
drone.connect()
drone.calibrate()

drone.take_off(3)
drone.land(2)
drone.stop()
