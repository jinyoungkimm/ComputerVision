from pixellib.instance import instance_segmentation
import cv2 as cv

capture = cv.VideoCapture(0)

seg_video = instance_segmentation()
seg_video.load_model("./pixellib(mask_RCNN)/mask_rcnn_coco.h5")


# n개의 클래스 중 관심 부류(target class)를 [person,book], 이 2개로 한정하여 instance segmentation한다
target_class = seg_video.select_target_classes(person=True,book=True)
seg_video.process_camera(capture,segment_target_classes=target_class, frames_per_second=4, show_frames=True, frame_name='Pixellib',
                         show_bboxes = True)

capture.release()
cv.destroyAllWindows()
