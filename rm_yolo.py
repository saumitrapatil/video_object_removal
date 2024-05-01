import cv2
import os
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt


def detect_objects(frame, model):
    results = model.predict(source=frame, save_txt=True)
    return results


def generate_mask(results, frame_width, frame_height):
    if results is None:
        return None

    masks = results[0].masks.data
    if masks is None or len(masks) < 2:
        return None

    mask = (masks[2].cpu().numpy() * 255).astype("uint8")
    mask = cv2.resize(mask, (frame_width, frame_height))
    return mask


def inpaint_frame(frame, mask, background):
    inpainted_with_background = frame.copy()
    inpainted_with_background[mask != 0] = background[mask != 0]
    return inpainted_with_background


def inpaint_with_masks_and_background(
    video_path,
    background_path,
    output_folder,
    method=cv2.INPAINT_TELEA,
    output_codec="XVID",
    frame_rate=30,
):
    model = YOLO("models/yolov8x-seg.pt")
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    background = cv2.imread(background_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_width = frame_width * 3

    out_input = cv2.VideoWriter(
        os.path.join(output_folder, "InputVideo.mp4"),
        cv2.VideoWriter_fourcc(*output_codec),
        frame_rate,
        (frame_width, frame_height),
    )
    out_masked_input = cv2.VideoWriter(
        os.path.join(output_folder, "MaskedInputVideo.mp4"),
        cv2.VideoWriter_fourcc(*output_codec),
        frame_rate,
        (frame_width, frame_height),
    )
    out_inpaint = cv2.VideoWriter(
        os.path.join(output_folder, "InpaintedVideo.mp4"),
        cv2.VideoWriter_fourcc(*output_codec),
        frame_rate,
        (frame_width * 3, frame_height),
    )

    for frame_number in range(frame_count):
        success, frame = cap.read()
        if not success:
            print(
                f"Error reading frame {frame_number} from video {video_path}. Skipping this frame."
            )
            continue

        results = detect_objects(frame, model)
        mask = generate_mask(results, frame_width, frame_height)

        if mask is None:
            print(f"No masks found for frame {frame_number}. Skipping this frame.")
            continue

        inpainted_with_background = inpaint_frame(frame, mask, background)

        combined_frame = np.concatenate(
            (frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), inpainted_with_background),
            axis=1,
        )

        out_input.write(frame)
        out_masked_input.write(cv2.bitwise_and(frame, frame, mask=mask))
        out_inpaint.write(inpainted_with_background)

        combined_frame = cv2.putText(
            combined_frame,
            "Input Video",
            (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        combined_frame = cv2.putText(
            combined_frame,
            "Masked Input Video",
            (frame_width + 10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        combined_frame = cv2.putText(
            combined_frame,
            "Inpainted Video",
            (2 * frame_width + 10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        combined_frame_resized = cv2.resize(combined_frame, (total_width, frame_height))

        cv2.namedWindow("Input, Mask, and Inpainted Videos", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input, Mask, and Inpainted Videos", total_width, frame_height)
        cv2.imshow("Input, Mask, and Inpainted Videos", combined_frame_resized)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    out_input.release()
    out_masked_input.release()
    out_inpaint.release()
    cv2.destroyAllWindows()


def extract_background_image(video_path, output_image_path, num_frames=30):
    # Open the video
    video = cv2.VideoCapture(video_path)

    # Get total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Randomly select frame indices
    FOI = np.random.randint(0, total_frames, size=num_frames)

    frames = []
    for frameOI in FOI:
        # Set the video capture to the selected frame index
        video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        # Read the frame
        ret, frame = video.read()
        if ret:  # Check if the frame is read successfully
            frames.append(frame)
        else:
            print("Error reading frame at index:", frameOI)

    # Calculate the median of the frames
    background_frame = np.median(frames, axis=0).astype(np.uint8)

    # Display the background frame
    plt.imshow(background_frame)
    plt.show()

    # Save the background image
    cv2.imwrite(output_image_path, background_frame)


def main():
    video_path = "InputVideos/test_video.mp4"
    background_path = "Background/backgroundImage.jpg"
    output_folder = "OutputVideos"

    # Extract Background
    extract_background_image(video_path, background_path, num_frames=30)

    # Apply Object detection and inpaint
    inpaint_with_masks_and_background(
        video_path,
        background_path,
        output_folder,
        method=cv2.INPAINT_TELEA,
        output_codec="mp4v",
        frame_rate=30,
    )


if __name__ == "__main__":
    main()
