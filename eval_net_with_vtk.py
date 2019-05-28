import numpy as np
import os
import tensorflow as tf
import Model
import cv2
from Util.util import loadObj, writeObj
import vtk
from Util.util import GetFaceMapper
from vtk.util.colors import *
import time
mean_value = 113.48202195680676

y = [0]
f = [0]


def img2tfimge(img):
    img = img[:, :, np.newaxis]
    img = img.reshape(-1, 258, 386, 1)
    x = (img.astype(np.float32) - mean_value) / 255
    return x


class MyInteractor(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, actor, renderwindow, sess, img_path, X, output, parent=None):
        self.timer_count = 0
        self.renderwindow = renderwindow
        self.actor = actor
        self.sess = sess
        self.X = X
        self.output = output
        self.img_path = img_path
        self.AddObserver("CharEvent", self.OnCharEvent)
        self.AddObserver("KeyPressEvent", self.OnKeyPressEvent)
        self.AddObserver("TimerEvent", self.execute)
        self.i = 0
        self.win = cv2.namedWindow("image")

    def OnCharEvent(self, obj, event):
        pass

    def OnKeyPressEvent(self, obj, event):
        # key = self.GetInteractor().GetKeySym()
        if (key == "Right"):
            print("hhh")
            pass

        if (key == "Up"):
            pass

        if (key == "Down"):
            pass

    def execute(self, obj, event):
        # print("y is {}".format(y))
        # print("f is {}".format(f))
        if len(f) == 1:
            print("the variable is none")
            pass
        else:
            img_file = os.path.join(self.img_path, "{}.jpg".format(self.i))
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.T  # 取决于图片是不是倒着的
            img = cv2.resize(img, dsize=(258, 386))
            cv2.imshow("image", img)
            img = img2tfimge(img)
            y = self.sess.run(self.output, feed_dict={self.X: img}) / 1000.0
            y = y.reshape((-1, 3))
            mapper = GetFaceMapper(y, f)
            self.actor.SetMapper(mapper)
            self.renderwindow.Render()
            self.i += 1
            if self.i % 10 == 0:
                print("show {}".format(self.i))
                # writeObj(os.path.join(test_output_path, "{}.obj".format(self.i)), y.tolist(), f)
        return


def LoadModel(filename):
    v, f = loadObj(filename)
    mapper = GetFaceMapper(v, f)
    actor = vtk.vtkLODActor()
    actor.SetMapper(mapper)
    return actor  # represents an entity in a rendered scene


def CreateCoordinates():
    # create coordinate axes in the render window
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(50, 50, 50)  # Set the total length of the axes in 3 dimensions

    # Set the type of the shaft to a cylinder:0, line:1, or user defined geometry.
    axes.SetShaftType(0)
    tprop = vtk.vtkTextProperty()
    tprop.SetFontSize(1)  # seems to be overriden by vtkCaptionActor2D
    # tprop.SetBold(1)
    # tprop.SetItalic(0)
    # tprop.SetColor(1.0, 1.0, 1.0)
    # tprop.SetOpacity(1.0)
    # tprop.SetFontFamilyToTimes()

    axes.SetCylinderRadius(0.02)
    # axes.GetXAxisCaptionActor2D().SetFontSize(10)
    # axes.GetYAxisCaptionActor2D().GetTextProperty().SetFontSize(10)
    # axes.GetZAxisCaptionActor2D().GetTextProperty().SetFontSize(10)
    axes.GetXAxisCaptionActor2D().SetWidth(0.03)
    axes.GetYAxisCaptionActor2D().SetWidth(0.03)
    axes.GetZAxisCaptionActor2D().SetWidth(0.03)
    for label in [
        axes.GetXAxisCaptionActor2D(),
        axes.GetYAxisCaptionActor2D(),
        axes.GetZAxisCaptionActor2D(),
    ]:

        label.SetCaptionTextProperty(tprop)
    # axes.SetAxisLabels(0)  # Enable:1/disable:0 drawing the axis labels
    # transform = vtk.vtkTransform()
    # transform.Translate(0.0, 0.0, 0.0)
    # axes.SetUserTransform(transform)
    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1,0,0)
    # axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().BoldOff() # disable text bolding
    return axes


def CreateGround():
    # create plane source
    plane = vtk.vtkPlaneSource()
    plane.SetXResolution(50)
    plane.SetYResolution(50)
    plane.SetCenter(0, 0, 0)
    plane.SetNormal(0, 0, 1)
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    # actor.GetProperty().SetOpacity(0.4)  # 1.0 is totally opaque and 0.0 is completely transparent
    actor.GetProperty().SetColor(light_grey)
    transform = vtk.vtkTransform()
    transform.Scale(2000, 2000, 1)
    actor.SetUserTransform(transform)
    return actor


def CreateScene(file, sess, img_path, X, output):
    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.Render()
    renWin.SetWindowName("visulization")

    # Create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    actor = LoadModel(file)  # load model
    r = vtk.vtkMath.Random(.4, 1.0)
    g = vtk.vtkMath.Random(.4, 1.0)
    b = vtk.vtkMath.Random(.4, 1.0)
    actor.GetProperty().SetDiffuseColor(r, g, b)
    actor.GetProperty().SetDiffuse(.8)
    actor.GetProperty().SetSpecular(.5)
    actor.GetProperty().SetSpecularColor(1.0, 1.0, 1.0)
    actor.GetProperty().SetSpecularPower(30.0)

    style = MyInteractor(actor=actor, renderwindow=renWin, sess=sess, img_path=img_path, X=X, output=output)
    style.SetDefaultRenderer(ren)
    iren.SetInteractorStyle(style)
    transform = vtk.vtkTransform()
    transform.Scale(500, 500, 500)
    transform.RotateX(90)
    actor.SetUserTransform(transform)
    actor.SetPosition(0, 0, 0)

    ren.AddActor(actor)
    ground = CreateGround()
    ren.AddActor(ground)

    ren.SetBackground(.2, .2, .2)

    renWin.SetSize(900, 600)  # width height

    # Set up the camera to get a particular view of the scene
    camera = vtk.vtkCamera()
    camera.SetViewAngle(30)
    camera.SetFocalPoint(300, 0, 0)
    camera.SetPosition(300, -400, 350)
    camera.ComputeViewPlaneNormal()
    camera.SetViewUp(0, 0, 0)
    camera.Zoom(0.4)
    ren.SetActiveCamera(camera)

    iren.Initialize()
    iren.CreateRepeatingTimer(20)  # the position is important
    iren.Start()  # 这行代码会阻塞事件
    # time.sleep(1)


if __name__ == '__main__':
    img_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\image\\048170110027"
    obj_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\smooth"
    model_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\net_model\\net_model-20000"
    test_output_path = "\\\\192.168.20.63\\ai\\face_data\\20190419\\Data_DeepLearning\\test_output"
    X = tf.placeholder(shape=[None, 258, 386, 1], dtype=tf.float32, name='input_x')
    model = Model.Face_net(X)
    output = model.face_net()
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    _, f = loadObj(os.path.join(obj_path, "smooth-0.obj"))


    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        first_time = True
        CreateScene(os.path.join(obj_path, "smooth-0.obj"), sess, img_path, X, output)

        # writeObj(os.path.join(test_output_path, "{}.obj".format(i)), y.tolist(), f)