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
    bl_label = "Auto Key Retopo Operator"

    # p = our pre-treined model directory
    p = "C:\\Users\\IAAdmin\\Documents\\GitHub\\auto-keypoint-retopology\\shape_predictor_68_face_landmarks.dat"  # windows
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    _timer = None
    _cap = None

    width = 800
    height = 600

    stop: bpy.props.BoolProperty()

    # 3D model points.
    model_points = numpy.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=numpy.float32)
    # Camera internals
    camera_matrix = numpy.array(
        [[height, 0.0, width / 2],
         [0.0, height, height / 2],
         [0.0, 0.0, 1.0]], dtype=numpy.float32
    )

    # Keeps min and max values, then returns the value in a ranve 0 - 1
    def get_range(self, name, value):
        if not hasattr(self, 'range'):
            self.range = {}
        if not name in self.range:
            self.range[name] = numpy.array([value, value])
        else:
            self.range[name] = numpy.array([min(value, self.range[name][0]), max(value, self.range[name][1])])
        val_range = self.range[name][1] - self.range[name][0]
        if val_range != 0:
            return (value - self.range[name][0]) / val_range
        else:
            return 0

    def modal(self, context, event):
        if event.type == 'TIMER':
            self.init_camera()
            _, image = self._cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            # bpy.context.scene.frame_set(frame_num)

            # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 2D image points. If you change the image, you need to change vector
                image_points = numpy.array([shape[30],  # Nose tip - 31
                                            shape[8],  # Chin - 9
                                            shape[36],  # Left eye left corner - 37
                                            shape[45],  # Right eye right corne - 46
                                            shape[48],  # Left Mouth corner - 49
                                            shape[54]  # Right mouth corner - 55
                                            ], dtype=numpy.float32)

                dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion

                if hasattr(self, 'rotation_vector'):
                    (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points,
                                                                                            image_points,
                                                                                            self.camera_matrix,
                                                                                            dist_coeffs,
                                                                                            flags=cv2.SOLVEPNP_ITERATIVE,
                                                                                            rvec=self.rotation_vector,
                                                                                            tvec=self.translation_vector,
                                                                                            useExtrinsicGuess=True)
                else:
                    (success, self.rotation_vector, self.translation_vector) = cv2.solvePnP(self.model_points,
                                                                                            image_points,
                                                                                            self.camera_matrix,
                                                                                            dist_coeffs,
                                                                                            flags=cv2.SOLVEPNP_ITERATIVE,
                                                                                            useExtrinsicGuess=False)

                if not hasattr(self, 'first_angle'):
                    self.first_angle = numpy.copy(self.rotation_vector)

                bones = bpy.data.objects["vincent_blenrig"].pose.bones

                bones["head_fk"].rotation_euler[0] = self.smooth_value("h_x", 3, (
                        self.rotation_vector[0] - self.first_angle[0])) / 1  # Up/Down
                bones["head_fk"].rotation_euler[2] = self.smooth_value("h_y", 3, -(
                        self.rotation_vector[1] - self.first_angle[1])) / 1.5  # Rotate
                bones["head_fk"].rotation_euler[1] = self.smooth_value("h_z", 3, (
                        self.rotation_vector[2] - self.first_angle[2])) / 1.3  # Left/Right

                bones["mouth_ctrl"].location[2] = self.smooth_value("m_h", 2, -self.get_range("mouth_height",
                                                                                              numpy.linalg.norm(
                                                                                                  shape[62] - shape[
                                                                                                      66])) * 0.06)
                bones["mouth_ctrl"].location[0] = self.smooth_value("m_w", 2, (
                        self.get_range("mouth_width", numpy.linalg.norm(shape[54] - shape[48])) - 0.5) * -0.04)
                bones["brow_ctrl_L"].location[2] = self.smooth_value("b_l", 3, (
                        self.get_range("brow_left", numpy.linalg.norm(shape[19] - shape[27])) - 0.5) * 0.04)
                bones["brow_ctrl_R"].location[2] = self.smooth_value("b_r", 3, (
                        self.get_range("brow_right", numpy.linalg.norm(shape[24] - shape[27])) - 0.5) * 0.04)

                bones["head_fk"].keyframe_insert(data_path="rotation_euler", index=-1)
                bones["mouth_ctrl"].keyframe_insert(data_path="location", index=-1)
                bones["brow_ctrl_L"].keyframe_insert(data_path="location", index=2)
                bones["brow_ctrl_R"].keyframe_insert(data_path="location", index=2)

                for (x, y) in shape:
                    cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

            cv2.imshow("Output", image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}

    def execute(self, context):
        if len(bpy.context.selected_objects) == 0:
            print("no object selected!")
            return

        obj = bpy.context.selected_objects[0]

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None


def register():
    bpy.utils.register_class(AutoKeyRetopoOperator)


def unregister():
    bpy.utils.unregister_class(AutoKeyRetopoOperator)


if __name__ == "__main__":
    register()

    # test call
    # bpy.ops.wm.opencv_operator()
