from ultralytics import YOLO
import cv2
import os
from tkinter import Tk, filedialog


# -----------------------------------
# 1. Select an image from computer
# -----------------------------------

root = Tk()
root.withdraw()

print("=" * 50)
print("       YOLO OBJECT DETECTION")
print("=" * 50)

print("\nSelect an image...")

image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
        ("All Files", "*.*")
    ]
)

# Check whether user selected an image
if not image_path:
    print("\nNo image selected.")
    print("Program stopped.")
    exit()

print("\nSelected image:")
print(image_path)


# -----------------------------------
# 2. Load YOLO model
# -----------------------------------

print("\nLoading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")


# -----------------------------------
# 3. Detect objects
# -----------------------------------

print("\nDetecting objects...")

results = model(image_path)

result = results[0]


# -----------------------------------
# 4. Print detected objects
# -----------------------------------

print("\n" + "=" * 50)
print("              DETECTION RESULTS")
print("=" * 50)

object_count = {}

if len(result.boxes) == 0:
    print("\nNo objects detected.")

else:

    for box in result.boxes:

        # Get class ID
        class_id = int(box.cls[0])

        # Get confidence
        confidence = float(box.conf[0])

        # Get object name
        object_name = model.names[class_id]

        # Print detection
        print(
            f"Object: {object_name:<15} "
            f"Confidence: {confidence * 100:.2f}%"
        )

        # Count objects
        if object_name in object_count:
            object_count[object_name] += 1
        else:
            object_count[object_name] = 1


# -----------------------------------
# 5. Print object summary
# -----------------------------------

if object_count:

    print("\n" + "-" * 50)
    print("OBJECT SUMMARY")
    print("-" * 50)

    total_objects = 0

    for object_name, count in object_count.items():

        print(f"{object_name}: {count}")

        total_objects += count

    print("-" * 50)

    print(f"Total Objects Detected: {total_objects}")


# -----------------------------------
# 6. Draw bounding boxes
# -----------------------------------

detected_image = result.plot()


# -----------------------------------
# 7. Create results folder
# -----------------------------------

os.makedirs("results", exist_ok=True)


# -----------------------------------
# 8. Generate output filename
# -----------------------------------

original_filename = os.path.basename(image_path)

filename_without_extension = os.path.splitext(
    original_filename
)[0]

output_path = os.path.join(
    "results",
    filename_without_extension + "_detected.jpg"
)


# -----------------------------------
# 9. Save detected image
# -----------------------------------

cv2.imwrite(
    output_path,
    detected_image
)

print("\n" + "=" * 50)

print("Detection completed successfully!")

print("\nResult saved to:")

print(output_path)

print("=" * 50)


# -----------------------------------
# 10. Resize image for display
# -----------------------------------

height, width = detected_image.shape[:2]

max_width = 1200
max_height = 800

scale = min(
    max_width / width,
    max_height / height,
    1
)

new_width = int(width * scale)
new_height = int(height * scale)

display_image = cv2.resize(
    detected_image,
    (new_width, new_height)
)


# -----------------------------------
# 11. Display result
# -----------------------------------

cv2.imshow(
    "YOLO Object Detection Result",
    display_image
)

print("\nPress any key on the image window to close.")

cv2.waitKey(0)

cv2.destroyAllWindows()