import cv2
import sys

cap_1 = cv2.VideoCapture(sys.argv[1] + "_1.avi")
cap_2 = cv2.VideoCapture(sys.argv[1] + "_2.avi")

saved = 0

while True:
	# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
	global frame_1, frame_2
	ret_1, frame_1 = cap_1.read()
	ret_2, frame_2 = cap_2.read()

	if saved == 100:
		cv2.imwrite("frame_1.jpg", frame_1)
		cv2.imwrite("frame_2.jpg", frame_2)
	
	saved+=1

	cv2.imshow(sys.argv[1] + "_1", frame_1)
	cv2.imshow(sys.argv[1] + "_2", frame_2)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
print("saved", saved)
