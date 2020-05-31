# Face-Recognition using OpenCV:

This is a OpenCV implementation of face recognizer on live/saved video.
## Dataset:

Collected f.r.i.e.n.d.s dataset and saved under the folder named by their character name. You can use your own dataset.

## Usage:

- Step-1 : Run extract_embedding.py to get embeddings of faces in the dataset using openCV
- Step-2 : Run train_model.py to train the classifier model on the embeddings to recognize the face(person).
- Step-3 : Run recognize.py to get face detection on a image.
- Step-4 : Run recognize_video.py to get face recogniton on your video.
