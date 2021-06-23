## Description

This is the repository for the Suitceyes-Visual Analysis module. In visual analysis or VA, the incoming messages that contain live images from a remote camera are processed and analyzed. The analysis consists of 4 main areas, which are object detection and classification, face detection, classification and facemask detection and scene recognition. Various models were trained for these operations. The results from the aforementioned operations can be distributed through a messaging bus that handles the communication between the camera, the VA service and potentially any module that would require those results. These results consist of JSON files that describe the detected objects or human faces, their position and the class that they belong. In order to utilize the aforementioned platform capabilities of the module, a subscription to the said message bus (VA_KBS_channel) is required. The number of simultaneous subscriptions depends on the capabilities of the broker utilized.
## Requirements - Dependencies

The Visual analysis module is developed in Python version 3.5. Below the additional dependencies are listed. Also in the document requirements.txt the additional libraries that will have to be intalled or have an account created, are also listed.

[Ably framework](https://ably.com/ ): A a pub/sub messaging broker for web and mobile apps. Account is required.

[Tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection): An open source framework built on top of TensorFlow that helps the construction, training and deployment of object detection models.

[Facenet library](https://github.com/davidsandberg/facenet): A TensorFlow implementation of the face recognizer.

[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit ): Α development environment for creating high performance GPU-accelerated applications.


## Instructions

1. Download and install the Tensorflow Object Detection Api. You can see [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md ) detailed instructions.
2. Clone the project to your local directory.
3. [Download](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md )  the frozen graph for object detection and place in the models/oid_objects directory. The openimages based models are compatible. 
4. [Download](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) and place in the models/20180402-114759 directory, the pretrained model for face detection.
5. Download and install Lampp,
6. Create a file where the images will be uploaded and stored.
8. Download and install CUDA in addition to the Python libraries listed in the requirements.txt document.
9 Start lampp service and run python3 listener.py.

## Platform Capabilities

To utilize the module as a platform and obtain the VA results that include the JSON format output, subscribe to the VA_KBS_channel provided by the Ably broker. The output of the VA analysis contains basic information about the image (dimensions, timestamp and name), details about the objects that are detected, which are: the type of object with the corresponding confidence and its relative position in the image  the human faces, their positions which are also recognised with the assistance of the Facenet library and if they wear facemask. Also, the scene recognition output is included.

## Trained Models
For object detection, the OpenImages(Kuznetsova, Alina, et al., 2020) dataset has been used for the development of 2 models, the main one that is computationally heavier and a lighter one that can be placed on mobile devices. For scene recognition the Places(Zhou, Bolei, et al.,2017) dataset, has been utilized, for the development of 2 versions, the main one and a lighter, for the same reason as object detection. For facemask detection the [Real-World Masked Face] (https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) and [Simulated Masked Face] (https://github.com/prajnasb/observations) datasets have been utilized. For access to the developed models contact Elias Kouslis (information below).

## Standalone Version
In the folder standalone version, a version of the Visual analysis is stored, which is meant to be executed on the mobile device, without needing an external processing device. It has been tested on a Raspberry Pi 4 Model B 4GB. To run, install the requirements and replace the listener and analyzer files with the files in the standalone version folder and start the standalone_va service. Lighter object and scene detection and recognition models have been developed for this version due to the limited capabilities of the raspberry device.

## Citations
Kuznetsova, Alina, et al. "The open images dataset v4." International Journal of Computer Vision (2020): 1-26
Zhou, Bolei, et al. "Places: A 10 million image database for scene recognition." IEEE transactions on pattern analysis and machine intelligence 40.6 (2017): 1452-1464.
## Contact 

For further details, please contact Elias Kouslis (kouslis@iti.gr).

