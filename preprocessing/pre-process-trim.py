import cv2
import mediapipe as mp
import os

# Function to calculate hand bounding box coordinates from landmarks with padding
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
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Load the video
video_path = "E:/signs/NSL23/S3_NSL_Consonant_Prepared/S3_Consonant_Camera.MOV"
output_folder = "E:/major-project/pre-images/pre-for-ka"
frame_skip_interval = 5
start_time = 2  
end_time = 4.5

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# Convert start and end time to frame numbers
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

frame_count = 0
file_counter = 137

while True:
    # Read the next frame
    ret, frame = cap.read()
    
    # Break the loop if the video is over or we reach the end time
    if not ret or frame_count > end_frame:
        break
    
    # Apply frame subsampling
    if frame_count >= start_frame and frame_count % frame_skip_interval == 0:
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands in the frame
        results = hands_detector.process(frame_rgb)
        
        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Access the hand landmarks (keypoints) to get the bounding box coordinates
                x_min, x_max, y_min, y_max = get_hand_bounding_box(frame, hand_landmarks)
                
                # Crop the hand region and save it as an image
                hand_image = frame[y_min:y_max, x_min:x_max]
                output_file = os.path.join(output_folder, f"hand_{file_counter:04d}.png")
                cv2.imwrite(output_file, hand_image)
                
                file_counter += 1  # Increment the counter for filenames
    
    frame_count += 1

# Release the video capture object
cap.release()
