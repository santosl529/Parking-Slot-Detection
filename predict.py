from ultralytics import YOLO
import torch
import cv2
import os
import time
import numpy as np
import glob
import argparse

# Save start time
start_time = time.time()

# Load the model
model = YOLO('runs/detect/train/weights/best.pt')

#Video Reader Array
valid_images = []

def run_model(source, confidence, isVideo): 
    "Process images/video and plug into model"
    images = []
    if isVideo:
        results = model(source, conf=confidence)
        return results
    else:
        for file in os.listdir(source):
            images.append(os.path.join(source, file))
        print(f"Images found: {images}")
        results = model(images, conf=confidence)
        return results

def filter_bboxes(result, width_multiplier):
    """Sort bounding boxes by smallest x coordinate in ascending order
    Filter out small boxes (Must be bigger than width_multiplier * average bounding box area)
    """
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()  # Extract confidences
    num_of_boxes = len(boxes_xyxy)
    if num_of_boxes != 0:
        boxes_xyxy = boxes_xyxy[boxes_xyxy[:, 0].argsort()]  # Sort by x-coordinate
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        avg_area = np.mean(areas)

        # Filter out small bboxes
        valid_boxes = boxes_xyxy[areas >= width_multiplier * avg_area]
        valid_confidences = confidences[areas >= width_multiplier * avg_area]

        num_of_boxes = len(valid_boxes)
        widths = valid_boxes[:, 2] - valid_boxes[:, 0]
        avg_width = np.mean(widths)
        return avg_width, valid_boxes.tolist(), valid_confidences.tolist(), num_of_boxes
    return None

def find_spaces(num_of_boxes, boxes_final, avg_width, camera_multiplier):
    """Locate spaces between detected cars
    Space must be greater than avg_width after applying the camera multiplier
    """
    num_of_spaces = 0
    space_coords = []
    avg_width *= camera_multiplier # Account for angle of camera and bounding box orientation
    for i in range(num_of_boxes - 1):
        x1 = boxes_final[i + 1][0]
        x2 = boxes_final[i][2]
        if (x1 - x2) >= avg_width:
            y = (boxes_final[i + 1][3] + boxes_final[i][3]) / 2
            num_of_spaces += 1
            space_coords.append((x1, x2, int(y)))
    return space_coords, num_of_spaces

def draw_boxes(image, boxes, confidences):
    for box, conf in zip(boxes, confidences):
        start_point = (int(box[0]), int(box[1]))  # Top-left corner
        end_point = (int(box[2]), int(box[3]))  # Bottom-right corner
        color = (0, 255, 0)  # Green color for bounding boxes
        thickness = 2  # Thickness of the bounding box lines
        cv2.rectangle(image, start_point, end_point, color, thickness)

        # Add confidence text
        text = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 4
        text_thickness = 8
        text_color = (0, 0, 0)
        text_background_color = (0, 255, 0)
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # Set the text start position
        text_org = (start_point[0], start_point[1] - 5)
        # Draw background rectangle for text
        cv2.rectangle(image, (text_org[0] - 1, text_org[1] - text_height - 1), 
                      (text_org[0] + text_width + 1, text_org[1] + 1), text_background_color, -1)
        # Put the text above the box
        cv2.putText(image, text, text_org, font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    return image

def draw_spaces(image, space_coords):
    """Draw Parking Spaces"""
    for slot in space_coords:
        start_point = (int(slot[0]), slot[2])  # (x, y)
        end_point = (int(slot[1]), slot[2])   # (x, y)
        color = (0, 0, 245)  # Red color for parking slot lines
        thickness = 20  # Thickness of the line
        cv2.line(image, start_point, end_point, color, thickness)
    return image

def process_result(result, counter, camera_multiplier, width_multiplier, isVideo):
    """Saves image into final result"""
    output = filter_bboxes(result, width_multiplier)
    if output:
        avg_width, boxes_final, confidences, num_of_boxes = output
        img = result.orig_img  # Access the original image from the result

        print(f"RESULT #{counter}")

        if num_of_boxes != 0:
            if isVideo:
                space_coords, num_of_spaces = find_spaces(num_of_boxes, boxes_final, avg_width, camera_multiplier)
                print(f"Number of Spaces: {num_of_spaces}")
                img_with_boxes = draw_boxes(img, boxes_final, confidences)
                img_with_lines = draw_spaces(img_with_boxes, space_coords)
                valid_images.append(img_with_lines)

                # Save the image with the bounding boxes and the lines
                # cv2.imwrite(os.path.join('videoResults', f'final_result{counter}.jpg'), img_with_lines)

            else:
                space_coords, num_of_spaces = find_spaces(num_of_boxes, boxes_final, avg_width, camera_multiplier)
                print(f"Number of Spaces: {num_of_spaces}")
                img_with_boxes = draw_boxes(img, boxes_final, confidences)
                img_with_lines = draw_spaces(img_with_boxes, space_coords)
                # Save the image with the bounding boxes and the lines
                cv2.imwrite(os.path.join('imageResults', f'final_result{counter}.jpg'), img_with_lines)

                print(f"Saved to imageResults as final_result{counter}.jpg")

def assemble_video(framerate, destination):
    first_image = valid_images[0]
    h, w, _ = first_image.shape
    total_images = len(valid_images)

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(destination, codec, framerate, (w, h))

    for counter, img in enumerate(valid_images):
        print("%.2f" % ((counter/total_images) * 100) + "%")
        loaded_img = img
        vid_writer.write(loaded_img)
            
    print("100.00%")
    vid_writer.release()


def main():
    parser = argparse.ArgumentParser(description="Parking Slot Detector")
    parser.add_argument("--cam_mult", 
                        help= "Changes the minimum space to be considered a parking space. > 1 increases space needed, < 1 increases space needed. Default is 0.9",
                        default=0.9)
    parser.add_argument("--conf", 
                        help= "Minimum model confidence when detecting cars. Default is 0.7",
                        default=0.7)
    parser.add_argument("-v",
                        help= "Sets source type to Video. Default is Image",
                        action="store_true",
                        )
    parser.add_argument("--size_mult", 
                        help= "Ignores cars in the background/distance. Higher number filters out more cars. Default is 0.7",
                        default=0.7)
    parser.add_argument("--src", 
                        help= "Path to images/videos. Default is /images or /videos based on the media type",
                        default="")
    parser.add_argument("--fr", 
                        help= "Video Framerate. Default is 30fps",
                        default=30)

    args = parser.parse_args()
    camera_multiplier = args.cam_mult # Reducing minimum space to be considered a parking space
    confidence = args.conf # Model minimum confidence rate
    isVideo = args.v # Is the source a video?
    width_multiplier = args.size_mult # Determines size minimum for bounding boxes
    if args.src == "": # Source of media to be processed
        if args.v:
            source = "videos"
        else:
            source = "images"
    results = run_model(source, confidence, isVideo)
    framerate = args.fr

    # Process results list
    for counter, result in enumerate(results):
        process_result(result, counter, camera_multiplier, width_multiplier, isVideo)
    if isVideo:
        number = len(glob.glob(os.path.join('videoResults', '*.mp4')))
        assemble_video(framerate, os.path.join('videoResults', f"final_result{number}.mp4"))
        print(f"Saved to videoResults as final_result{number}.mp4")

        # leftovers = glob.glob(os.path.join('videoResults', '*.jpg'))
        # for path in leftovers:
        #     os.remove(path)
        # print(f"Saved to videoResults as final_result{number}.mp4")
    

if __name__ == '__main__':
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
