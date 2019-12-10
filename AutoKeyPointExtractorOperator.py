# ugly fix if openvino is
import os
from functools import partial
from multiprocessing.pool import Pool

from mathutils import Vector

os.sys.path = list(filter(lambda x: "openvino" not in x, os.sys.path))

import bpy
from bpy_extras.object_utils import world_to_camera_view

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy import spatial

# settings
LANDMARK_PATH = "/Users/cansik/git/zhdk/auto-keypoint-retopology/shape_predictor_68_face_landmarks.dat"
RENDER_DIR = "/Users/cansik/git/zhdk/auto-keypoint-retopology/"


# mapping


def get_vertex(scene, cam, keypoint):
    co_2d = world_to_camera_view(scene, cam, keypoint)


def get_closest_vertex(keypoint, screen_coordinates):
    return screen_coordinates[0]


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
        i = 0
        for (x, y) in shape:
            cv2.putText(image, "%s" % i, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),
                        lineType=cv2.LINE_AA)
            cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
            i += 1

        cv2.imwrite(RENDER_DIR + "/result.png", image)
        #cv2.imshow("Output", image)
        #cv2.waitKey(1)
        return shape.tolist()

    def project_to_vertices(self, cam, keypoints):
        scene = bpy.context.scene
        return list(map(lambda kp: get_vertex(scene, cam, kp), keypoints))

    def get_screen_coordinates(self, scene, cam, obj):
        mat = obj.matrix_world

        # Multiply matrix by vertex
        vertices = (mat @ vert.co for vert in obj.data.vertices)
        return [world_to_camera_view(scene, cam, coord) for coord in vertices]

    def scale_to_pixel(self, scene, screen_coordinates):
        render_scale = scene.render.resolution_percentage / 100
        render_size = (
            int(scene.render.resolution_x * render_scale),
            int(scene.render.resolution_y * render_scale),
        )
        return [list((round(v[0] * render_size[0]), round(v[1] * render_size[1]))) for v in screen_coordinates]

    def execute(self, context):
        # get object to be annotated
        if len(bpy.context.selected_objects) == 0:
            print("no object selected!")
            return {'FINISHED'}

        # read objects
        scene = bpy.context.scene
        obj = bpy.context.selected_objects[0]
        cam = bpy.data.objects['Camera']

        # create render
        image_path = RENDER_DIR + "/render.png"
        self.render_to_file(image_path)

        # extract keypoints
        keypoints = self.extract_keypoints(image_path)

        # extract vertices
        screen_coordinates = self.get_screen_coordinates(scene, cam, obj)

        # create list but only take x and y as list (kp are 2d)
        screen_coordinate_list = [list(v[:2]) for v in screen_coordinates]
        scaled_screen_coordinates_list = self.scale_to_pixel(scene, screen_coordinate_list)

        # print("SCL: %s" % list(screen_coordinate_list[:5]))
        # print("SSCL: %s" % list(scaled_screen_coordinates_list[:5]))
        
        tree = spatial.KDTree(scaled_screen_coordinates_list)

        # match screen coordinates to keypoint positions
        # todo: filter points which are too fare away (otherwise backpoints match better)
        vertex_indexes = [tree.query(kp) for kp in keypoints]
        mean_accuracy = np.mean(vertex_indexes, axis=0)
        
        # extract real vertices
        vertices = [obj.data.vertices[vi[1]].co for vi in vertex_indexes]
        world_vertices = list(obj.matrix_world @ vert for vert in vertices)

        # add cubes for each vertex
        for i, v in enumerate(world_vertices[:30]):
            bpy.context.scene.cursor.location = (v.x, v.y, v.z)
            #bpy.ops.mesh.primitive_cube_add(location=(v.x, v.y, v.z), size=2)
            #bpy.ops.transform.resize(value=(0.1, 0.1, 0.1))

        print("-----")
        print("Points Extracted: %s pts" % len(vertex_indexes))
        print("Mean Accuracy: %s px" % round(mean_accuracy[0], 4))
        print("-----")

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
