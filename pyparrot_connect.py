from pyparrot.Minidrone import Mambo
import time


mamboAddr = "d0:3a:4d:9c:e6:24"

mambo = Mambo(mamboAddr, use_wifi=False)

print("Trying to connect:")
success = mambo.connect(num_retries=3)
print("Connected:", success)

if(success):
	print("Success!")

	print("Taking Off")
	mambo.safe_takeoff(5)

	print(mambo.sensors)

	mambo.smart_sleep(5)

	print("Landing")
	mambo.safe_land(5)

	mambo.smart_sleep(5)

print("Disconnecting")
mambo.disconnect()