# ugly fix if openvino is
import os

from mathutils import Vector

os.sys.path = list(filter(lambda x: "openvino" not in x, os.sys.path))

import bpy
from bpy_extras.object_utils import world_to_camera_view

import cv2
import dlib
import numpy
from imutils import face_utils

# settings
LANDMARK_PATH = "/Users/cansik/git/zhdk/auto-keypoint-retopology/shape_predictor_68_face_landmarks.dat"
RENDER_DIR = "/Users/cansik/git/zhdk/auto-keypoint-retopology/"


# mapping


def get_vertex(scene, cam, keypoint):
    co_2d = world_to_camera_view(scene, cam, keypoint)


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

        # find land marks for first face
        rect = self.detector(gray, 0)[0]
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # annotate keypoints in image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

        cv2.imwrite(RENDER_DIR + "/result.png", image)
        cv2.imshow("Output", image)
        cv2.waitKey(1)
        return shape.tolist()

    def project_to_vertices(self, cam, keypoints):
        scene = bpy.context.scene
        return list(map(lambda kp: get_vertex(scene, cam, kp), keypoints))

    def execute(self, context):
        # get object to be annotated
        if len(bpy.context.selected_objects) == 0:
            print("no object selected!")
            return {'FINISHED'}

        obj = bpy.context.selected_objects[0]
        cam = bpy.data.objects['Camera']

        # create render
        image_path = RENDER_DIR + "/render.png"
        self.render_to_file(image_path)

        # extract keypoints
        keypoints = self.extract_keypoints(image_path)
        positions = list(map(lambda k: Vector((k[0], k[1], 0.0)), keypoints))
        print("Positions: %s" % positions)

        # map keypoints to vertices
        vertices = self.project_to_vertices(cam, positions)
        print("Vertices: %s" % vertices)

        return {'FINISHED'}

    def cancel(self, context):
        cv2.destroyAllWindows()


def register():
    bpy.utils.register_class(AutoKeyPointExtractorOperator)


def unregister():
    bpy.utils.unregister_class(AutoKeyPointExtractorOperator)


if __name__ == "__main__":
    register()

    # test call
    bpy.ops.wm.auto_key_point_extractor_operator()
