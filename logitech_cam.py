import cv2

cap_1 = cv2.VideoCapture(1)
cap_2 = cv2.VideoCapture(2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_1 = cv2.VideoWriter('output_1.avi', fourcc, 20.0, (640,480))
out_2 = cv2.VideoWriter('output_2.avi', fourcc, 20.0, (640,480))

while True:
	# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
	global frame_1, frame_2
	ret_1, frame_1 = cap_1.read()
	ret_2, frame_2 = cap_2.read()

	cv2.imshow("capture_1", frame_1)
	cv2.imshow("capture_2", frame_2)

	out_1.write(frame_1)
	out_2.write(frame_2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
