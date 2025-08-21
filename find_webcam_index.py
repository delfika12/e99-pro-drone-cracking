import cv2

def find_camera_indices(max_indices=10):
    """
    Finds available camera indices by checking if they can be opened.

    Parameters:
    - max_indices (int): The maximum number of camera indices to check.

    Returns:
    - List[int]: A list of available camera indices.
    """
    available_cameras = []
    
    for index in range(max_indices):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            available_cameras.append(index)
        cap.release()
    
    return available_cameras

# Define the maximum number of indices to check
MAX_INDICES = 10

# Find and print available camera indices
available_cameras = find_camera_indices(MAX_INDICES)
if available_cameras:
    print("Available camera indices:", available_cameras)
else:
    print("No cameras found.")
