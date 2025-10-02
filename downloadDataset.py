from roboflow import Roboflow

rf = Roboflow(api_key="hUaMASK3d1LT1z3y8as3")
project = rf.workspace("realsoftai").project("mol-bozor-person-hujun")
version = project.version(3)
dataset = version.download("yolov11")

