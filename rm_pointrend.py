import torch, torchvision
import os

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# import PointRend project
from detectron2.projects import point_rend
from matplotlib import pyplot as plt

coco_metadata = MetadataCatalog.get("coco_2017_val")


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


def generate_mask(frame, frame_width, frame_height):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(
        "detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    predictor = DefaultPredictor(cfg)

    outputs = predictor(frame)
    mask = (outputs["instances"].pred_masks[-1].cpu().numpy() * 255).astype("uint8")
    mask = cv2.resize(mask, (frame_width, frame_height))
    return mask


def inpaint_frame(frame, mask, background):
    inpainted_with_background = frame.copy()
    inpainted_with_background[mask != 0] = background[mask != 0]
    return inpainted_with_background


def inpaint_with_mask_and_background(
    video_path,
    background_image_path,
    output_path,
    method=cv2.INPAINT_TELEA,
    output_codec="XVID",
    frame_rate=30,
):
    cap = cv2.VideoCapture(video_path)
    background = cv2.imread(background_image_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_width = frame_width * 3

    out_input = cv2.VideoWriter(
        os.path.join(output_path, "input_video.mp4"),
        cv2.VideoWriter_fourcc(*output_codec),
        frame_rate,
        (frame_width, frame_height),
    )
    out_masked_input = cv2.VideoWriter(
        os.path.join(output_path, "masked_input_video.mp4"),
        cv2.VideoWriter_fourcc(*output_codec),
        frame_rate,
        (frame_width, frame_height),
    )
    out_inpaint = cv2.VideoWriter(
        os.path.join(output_path, "inpainted_video.mp4"),
        cv2.VideoWriter_fourcc(*output_codec),
        frame_rate,
        (frame_width, frame_height),
    )

    for frame_number in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        mask = generate_mask(frame, frame_width, frame_height)
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


video_path = "InputVideos/test_video.mp4"
background_path = "Background/backgroundImage.jpg"
output_folder = "OutputVideos"

# Extract Background
extract_background_image(video_path, background_path, num_frames=30)

# Apply Object detection and inpaint
inpaint_with_mask_and_background(
    video_path,
    background_path,
    output_folder,
    method=cv2.INPAINT_TELEA,
    output_codec="XVID",
    frame_rate=30,
)
