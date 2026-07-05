# Eye Color Changer
A small python script to change the eye color of your AI characters.
It is not perfect but it is easier and faster than photoshoping it.

<img width="1200" height="600" alt="combined_0_185859" src="https://github.com/user-attachments/assets/dbc75509-fe9a-4e06-8f31-fecc0fe3e1a4" />


## INSTALL:
1- Download Eye [Color Changer.py](https://github.com/masterofobzene/eyecolorchanger/blob/main/Eye%20Color%20Changer.py) - (right click -> save link as)

2- Install python requirements. Open CMD and type: `pip install mediapipe==0.10.14` then `pip install opencv-python==4.10.0.84` and finally `pip install numpy`

3- Download [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt) and [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel). Put them in a final folder, then map them inside your script (step 4)

4- Change these lines with the correct path to your downloaded files in step 3:
````python
PROTO_TXT = os.path.join(SCRIPT_DIR, r"C:\YOUR_PATH_TO\deploy.prototxt")
CAFFE_MODEL = os.path.join(SCRIPT_DIR, r"C:\YOUR_PATH_TO\res10_300x300_ssd_iter_140000.caffemodel")
````


# USE:
`python "C:\whatever\Color Changer.py" img1.jpg img2.png img3.jpeg teal`

You can use other colors:

- blue
- teal
- green
- brown
- hazel
- gray
- purple

The script will excecute and put the new recolored versions of the images in the root where the originals were, and the originals will be put into a `ORIGINAL_EYES` folder. 

> [!TIP]
NOTE: if the script cannot detect the eyes on your pic (this can happen if the image is too small or the eyes are not very clear) the script will NOT move that file to the originals folder and will not do anything to it.

