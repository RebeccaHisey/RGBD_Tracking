import os
import unittest
import logging
import numpy
import pandas
import cv2

import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# US_Probe_Tracking
#

class US_Probe_Tracking(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "US_Probe_Tracking"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["RGB-D Tracking"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#US_Probe_Tracking">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # US_Probe_Tracking1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='US_Probe_Tracking',
    sampleName='US_Probe_Tracking1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'US_Probe_Tracking1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='US_Probe_Tracking1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='US_Probe_Tracking1'
  )

  # US_Probe_Tracking2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='US_Probe_Tracking',
    sampleName='US_Probe_Tracking2',
    thumbnailFileName=os.path.join(iconsPath, 'US_Probe_Tracking2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='US_Probe_Tracking2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='US_Probe_Tracking2'
  )

#
# US_Probe_TrackingWidget
#

class US_Probe_TrackingWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/US_Probe_Tracking.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    self.ui.videoIDComboBox.addItem("Select video ID")

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = US_Probe_TrackingLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.DirectoryButton.connect('directorySelected(QString)',self.updateParameterNodeFromGUI)
    self.ui.videoIDComboBox.connect('currentIndexChanged(int)', self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.DirectoryButton.connect('directorySelected(QString)',self.onDatasetSelected)
    self.ui.startTrackingButton.connect('clicked(bool)',self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def onDatasetSelected(self):
      for i in range(self.ui.videoIDComboBox.count,0,-1):
        self.ui.videoIDComboBox.removeItem(i)
      self.currentDatasetName = os.path.basename(self.ui.DirectoryButton.directory)
      self.videoPath = self.ui.DirectoryButton.directory
      self.addVideoIDsToComboBox()

  def addVideoIDsToComboBox(self):
    """
    when a new video ID is created, add it to the combo box
    :return:
    """
    for i in range(1,self.ui.videoIDComboBox.count + 1):
      self.ui.videoIDComboBox.removeItem(i)
    videoIDList = os.listdir(self.videoPath)
    self.videoIDList = [dir for dir in videoIDList if dir.rfind(".") == -1] #get only directories
    self.ui.videoIDComboBox.addItems(self.videoIDList)

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())


  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.DirectoryButton.directory = self._parameterNode.GetParameter("Dataset")
    self.ui.videoIDComboBox.setCurrentText(self._parameterNode.GetParameter("Video_ID"))

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetParameter("Dataset", str(self.ui.DirectoryButton.directory))
    self._parameterNode.SetParameter("Video_ID", str(self.ui.videoIDComboBox.currentText))

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      if self.ui.startTrackingButton.text == "Start Tracking":
        self.logic.startTracking(self.videoPath,self.ui.videoIDComboBox.currentText)
        self.ui.startTrackingButton.setText("Stop Tracking")
      else:
        self.logic.stopTracking()
        self.ui.startTrackingButton.setText("Start Tracking")

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# US_Probe_TrackingLogic
#

class US_Probe_TrackingLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    pass

  def startTracking(self, datasetDirectory, videoID):
    """
    Start processing the depth images.
    Can be used without GUI widget.
    :param datasetDirectory: directory where dataset is stored
    :param videoID: ID of video being shown
    """
    self.fid2ModLogic = slicer.util.getModuleLogic("FiducialsToModelRegistration")
    seekWidget = slicer.util.mainWindow().findChildren("qMRMLSequenceBrowserSeekWidget")
    seekWidget = seekWidget[0]
    timeLabel = seekWidget.findChildren("QLabel")
    self.timeLabel = timeLabel[1]
    self.probeModel = slicer.util.getFirstNodeByName("Telemed")
    try:
      self.probeToDepth = slicer.util.getFirstNodeByName("ProbeToDepth")
      if self.probeToDepth == None:
        self.probeToDepth = slicer.vtkMRMLLinearTransformNode()
        self.probeToDepth.SetName("ProbeToDepth")
        slicer.mrmlScene.AddNode(self.probeToDepth)
    except slicer.MRMLNodeNotFoundException:
      self.probeToDepth = slicer.vtkMRMLLinearTransformNode()
      self.probeToDepth.SetName("ProbeToDepth")
      slicer.mrmlScene.AddNode(self.probeToDepth)
    self.probeModel.SetAndObserveTransformNodeID(self.probeToDepth.GetID())

    rgbLabelFilePath = os.path.join(datasetDirectory, videoID,
                                      "RGB_right/{}_RGB_right_Labels.csv".format(videoID))
    self.RGBLabelFile = pandas.read_csv(rgbLabelFilePath)
    self.RGBLabelFile["Tool bounding box"] = [eval(self.RGBLabelFile["Tool bounding box"][i]) for i in self.RGBLabelFile.index]
    self.RGBLabelFile["Time Recorded"] = [round(self.RGBLabelFile["Time Recorded"][i],2) for i in self.RGBLabelFile.index]

    self.depthNode = slicer.util.getFirstNodeByClassByName("vtkMRMLStreamingVolumeNode","Image1DEPTH_Image1DE")
    self.depthNodeObserver = self.depthNode.AddObserver(slicer.vtkMRMLStreamingVolumeNode.FrameModifiedEvent,self.exportNewPoints)

  def stopTracking(self):
    self.depthNode.RemoveObserver(self.depthNodeObserver)
    self.depthNodeObserver = None

  def exportNewPoints(self,caller,eventID):
    timeStamp = self.getCurrentTimeStamp()
    rgbImage = self.RGBLabelFile.loc[self.RGBLabelFile["Time Recorded"]==timeStamp]

    if not rgbImage.empty:
      bbox = self.RGBLabelFile["Tool bounding box"][rgbImage.index[0]]
      usBBox = [box for box in bbox if box["class"]=="ultrasound"]
      if len(usBBox)>0:
        self.usBBox = usBBox[0]
        self.getDepthImage()
        self.convertDepthToPoints()
        self.updateDepthToRASTransform()

  def updateDepthToRASTransform(self):
    self.fid2ModLogic.run(self.fiducialNode,self.probeModel,self.probeToDepth)
    self.probeToDepth.Inverse()
    slicer.mrmlScene.Modified()

  def convertDepthToPoints(self):
    try:
      self.fiducialNode = slicer.util.getNode("depthFiducials")
      self.fiducialNode.RemoveAllMarkups()
    except slicer.util.MRMLNodeNotFoundException:
      self.fiducialNode = slicer.vtkMRMLMarkupsFiducialNode()
      self.fiducialNode.SetName("depthFiducials")
      slicer.mrmlScene.AddNode(self.fiducialNode)
    max_depth = self.depthImage.max()
    min_depth = self.depthImage.min()
    imageShape = self.depthImage.shape
    fidAddedCount = 0
    for x in range(0,imageShape[0],5):
      for y in range(0,imageShape[1],5):
        if self.depthImage[x][y]>0:
          depthValue = max_depth - self.depthImage[x][y]
          self.fiducialNode.AddFiducialFromArray(numpy.array([x,y,depthValue]))
          fidAddedCount += 1


  def removeColorizing(self):
    imdata = self.getVtkImageDataAsOpenCVMat()
    imdata = cv2.flip(imdata,0)
    bboxImdata = imdata[int(self.usBBox["ymin"]):int(self.usBBox["ymax"]),
                 int(self.usBBox["xmin"]):int(self.usBBox["xmax"])]
    shape = bboxImdata.shape
    self.depthImage = numpy.array([[self.convertRGBtoD(j) for j in bboxImdata[i]] for i in range(shape[0])])

  def convertRGBtoD(self,pixel1):
    is_disparity = False
    min_depth = 0.16
    max_depth = 300.0
    min_disparity = 1.0 / max_depth
    max_disparity = 1.0 / min_depth
    r_value = float(pixel1[0])
    g_value = float(pixel1[1])
    b_value = float(pixel1[2])
    depthValue = 0
    if (b_value + g_value + r_value) < 255:
      hue_value = 0
    elif (r_value >= g_value and r_value >= b_value):
      if (g_value >= b_value):
        hue_value = g_value - b_value
      else:
        hue_value = (g_value - b_value) + 1529
    elif (g_value >= r_value and g_value >= b_value):
      hue_value = b_value - r_value + 510

    elif (b_value >= g_value and b_value >= r_value):
      hue_value = r_value - g_value + 1020

    if (hue_value > 0):
      if not is_disparity:
        z_value = ((min_depth + (max_depth - min_depth) * hue_value / 1529.0) + 0.5);
        depthValue = z_value
      else:
        disp_value = min_disparity + (max_disparity - min_disparity) * hue_value / 1529.0
        depthValue = ((1.0 / disp_value) / 1000 + 0.5)
    else:
      depthValue = 0
    return depthValue


  def getVtkImageDataAsOpenCVMat(self):
    cameraVolume = self.depthNode
    '''if cameraVolume.GetClassName() == "vtkMRMLStreamingVolumeNode":
      image = cameraVolume.GetFrameData()'''

    image = cameraVolume.GetImageData()
    shape = list(cameraVolume.GetImageData().GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    imageMat = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    return imageMat

  def getDepthImage(self):
    imdata = self.getVtkImageDataAsOpenCVMat()

    shape = imdata.shape
    if len(shape) > 2:
      self.removeColorizing()
    else:
      imdata = cv2.flip(imdata, 0)
      bboxImdata = imdata[int(self.usBBox["ymin"]):int(self.usBBox["ymax"]),
                   int(self.usBBox["xmin"]):int(self.usBBox["xmax"])]
      self.depthImage = numpy.array([[j for j in bboxImdata[i]] for i in range(shape[0])])
    thresh = cv2.threshold(self.depthImage.astype("uint8"), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    numlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    largestArea = 0
    bestLabel = -1
    for i in range(2):
      if stats[i][4] > largestArea:
        bestLabel = i
        largestArea = stats[i][4]
    bestArea = numpy.where(labels==bestLabel,self.depthImage,0)
    self.depthImage = numpy.zeros((imdata.shape[0],shape[1]))
    self.depthImage[int(self.usBBox["ymin"]):int(self.usBBox["ymax"]),
                   int(self.usBBox["xmin"]):int(self.usBBox["xmax"])] = bestArea.astype("uint8")


  def getCurrentTimeStamp(self):
    recordingTime = float(self.timeLabel.text)
    return recordingTime

#
# US_Probe_TrackingTest
#

class US_Probe_TrackingTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_US_Probe_Tracking1()

  def test_US_Probe_Tracking1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('US_Probe_Tracking1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = US_Probe_TrackingLogic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
