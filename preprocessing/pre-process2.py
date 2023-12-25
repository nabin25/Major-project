import cv2
import mediapipe as mp
import os

# Function to calculate hand bounding box coordinates from landmarks
def get_hand_bounding_box(frame, hand_landmarks, padding=20):
    x_min, x_max, y_min, y_max = frame.shape[1], 0, frame.shape[0], 0
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    
    # Add padding to the bounding box coordinates
    x_min = max(0, x_min - padding)
    x_max = min(frame.shape[1], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(frame.shape[0], y_max + padding)
    
    return x_min, x_max, y_min, y_max

# Initialize the hand tracking module
mp_hands = mp.solutions.hands

# Create a hands detector
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load the video
video_path = "E:/signs/NSL23/S1_NSL_consonent_Bright_Cropped/S1_KA.mov"
output_folder = "E:/major-project/new"
frame_skip_interval = 5

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

frame_count = 0
file_counter = 0

# Read the first frame
ret, prev_frame = cap.read()

while True:
    # Read the next frame
    ret, frame = cap.read()
    
    # Break the loop if the video is over
    if not ret:
        break
    
    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply frame differencing for background subtraction
    diff_frame = cv2.absdiff(frame, prev_frame)
    fg_mask = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)

    # Combine the original frame with the foreground mask
    frame_rgb_with_mask = cv2.bitwise_and(frame_rgb, frame_rgb, mask=fg_mask)

    # Display the original frame
    cv2.imshow("Original Frame", frame)

    # Display the frame after background subtraction
    cv2.imshow("Frame with Background Subtraction", frame_rgb_with_mask)

    # Detect hands in the frame with background subtraction
    results = hands_detector.process(frame_rgb_with_mask)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Access the hand landmarks (keypoints) to get the bounding box coordinates
            x_min, x_max, y_min, y_max = get_hand_bounding_box(frame, hand_landmarks)

            # Crop the hand region from the original frame
            hand_image = frame[y_min:y_max, x_min:x_max]
            output_file = os.path.join(output_folder, f"hand_{file_counter:04d}.png")
            cv2.imwrite(output_file, hand_image)
            file_counter += 1

    # Display the frame after hand detection
    cv2.imshow("Frame with Hand Detection", cv2.cvtColor(results.image, cv2.COLOR_RGB2BGR))

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release the video capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
