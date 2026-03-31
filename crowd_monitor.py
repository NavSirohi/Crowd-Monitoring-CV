import argparse
import cv2
import sys
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Crowd Monitoring System - Calculate foot count in highly congested areas.")
    parser.add_argument("--image", required=True, help="Path to the input image file (e.g., data/sample_crowd.jpg)")
    parser.add_argument("--output", required=True, help="Path to save the annotated output image (e.g., results/output.jpg)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO model weights (default: yolov8n.pt for maximum speed)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: The input image '{args.image}' could not be found.", file=sys.stderr)
        sys.exit(1)
        
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("Loading YOLOv8 model for person detection...")
    # Using yolov8n (nano) which is extremely fast and auto-downloads the weights (~6MB) onto the system
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error initializing YOLO model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load the image {args.image}. Ensure it's a valid format.", file=sys.stderr)
        sys.exit(1)

    # Perform detection. The target class '0' corresponds to "person" in the COCO dataset.
    print("Running detection...")
    results = model.predict(source=image, classes=[0], conf=args.conf, verbose=False)

    person_count = 0
    if len(results) > 0:
        result = results[0]
        person_count = len(result.boxes)
        
        # Plot overlays bounding boxes directly onto the image
        annotated_image = result.plot()
        
        # Add visual counter text to the top-left of the image
        cv2.putText(
            annotated_image,
            f"Real-Time Count: {person_count}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,
            cv2.LINE_AA
        )

        cv2.imwrite(args.output, annotated_image)
        print(f"[SUCCESS] Successfully saved the annotated visualization to: {args.output}")
    else:
        print("No detections made.")
        
    print("\n" + "="*40)
    print("           ANALYSIS REPORT           ")
    print("="*40)
    print(f"Total People Detected: {person_count}")
    print(f"Input Processed      : {args.image}")
    print(f"Output Saved         : {args.output}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
