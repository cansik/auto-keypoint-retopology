import bpy
from bpy.props import FloatProperty

from imutils import face_utils
import dlib
import cv2
import time
import numpy


class AutoKeyRetopoOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.opencv_operator"
    bl_label = "Auto Key Retopology Operator"

    # our pre-treined model directory
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    stop: bpy.props.BoolProperty()

    def extract(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        # find land marks for faces
        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 2D image points
            image_points = numpy.array([shape[30],  # Nose tip - 31
                                        shape[8],  # Chin - 9
                                        shape[36],  # Left eye left corner - 37
                                        shape[45],  # Right eye right corner - 46
                                        shape[48],  # Left Mouth corner - 49
                                        shape[54]  # Right mouth corner - 55
                                        ], dtype=numpy.float32)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        cv2.imshow("Output", image)
        cv2.waitKey(1)
        return shape

    def execute(self, context):
        if len(bpy.context.selected_objects) == 0:
            print("no object selected!")
            return

        obj = bpy.context.selected_objects[0]

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()


def register():
    bpy.utils.register_class(AutoKeyRetopoOperator)


def unregister():
    bpy.utils.unregister_class(AutoKeyRetopoOperator)


if __name__ == "__main__":
    register()

    # test call
    # bpy.ops.wm.opencv_operator()
