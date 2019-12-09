# ugly fix if openvino is
import os
os.sys.path = list(filter(lambda x: "openvino" not in x, os.sys.path))

import bpy
import cv2
import dlib
import numpy
from imutils import face_utils

# pre-trained model
LANDMARK_PATH = "/Users/cansik/git/zhdk/auto-keypoint-retopology/shape_predictor_68_face_landmarks.dat"
RENDER_DIR = "/Users/cansik/git/zhdk/auto-keypoint-retopology/"

class AutoKeyPointExtractorOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.auto_key_point_extractor_operator"
    bl_label = "Auto KeyPoint Extractor Operator"

    print(os.path.dirname(os.path.abspath(__file__)))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_PATH)

    stop: bpy.props.BoolProperty()

    def render_to_file(self, filename):
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces[0].shading.type = 'RENDERED'

        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = filename
        bpy.ops.render.render(use_viewport=True, write_still=True)

    def extract_keypoints(self, filename):
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
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

        cv2.imwrite(RENDER_DIR + "/result.png", image)
        cv2.imshow("Output", image)
        cv2.waitKey(1)
        return shape

    def project_to_vertices(self, keypoints):
        return None

    def execute(self, context):
        # get object to be annotated
        if len(bpy.context.selected_objects) == 0:
            print("no object selected!")
            return {'FINISHED'}

        obj = bpy.context.selected_objects[0]

        # create render
        imagePath = RENDER_DIR + "/render.png"
        self.render_to_file(imagePath)

        # extract keypoints
        self.extract_keypoints(imagePath)

        return {'FINISHED'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()


def register():
    bpy.utils.register_class(AutoKeyPointExtractorOperator)


def unregister():
    bpy.utils.unregister_class(AutoKeyPointExtractorOperator)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.wm.auto_key_point_extractor_operator()
