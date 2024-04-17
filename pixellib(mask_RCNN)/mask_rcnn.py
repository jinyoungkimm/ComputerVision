from pixellib.instance import instance_segmentation # 사례 분할
import cv2 as cv


seg = instance_segmentation()
seg.load_model("./pixellib(mask_RCNN)/mask_rcnn_coco.h5")


img_fname = './pixellib(mask_RCNN)/busy_street.jpg'
info,img_segmented = seg.segmentImage(img_fname,show_bboxes=True)


cv.imshow("Image segmentation overlayed",img_segmented)


cv.waitKey()
cv.destroyAllWindows()






