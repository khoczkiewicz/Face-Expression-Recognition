# Face-Expression-Recognition

Face expression recognition using fuzzy control system (scikit-fuzzy)

## Introduction

An app is designed to detect and recognize face expression (and partially expressions). It uses fuzzy logic to determine which emotion is portraited on the picture.

## Motivation

#### Amanda Patterson

Body-language's sheets described by _Amanda Patterson_ provides an information about specific "translations". It's best way to understand the purposes of the experiment and idea of image recognition shown below.

Example:

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/readme-images/Cheat-Sheets-For-Body-Language-Part-1.jpg)

#### Paul Ekman

Main character of tv-series _Lie to Me_ (played by an american actor _Tim Roth_) is psychologist being an extraoridinary specialist of emotions recognition based on facial expressions.

Example:

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/tim_roth_images/5.-Paul-Ekman-lie-detector-.jpg)

## Abstraction

The result of the experiment is described by `Tim Roth Coefficient`. This trully confusing name of the variable provides an information about which emotion is fuzzed by controler.

## Features

#### Facial landmarks predictor

```python
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

Enumerating through detections and using ready-made predictor (code above) gives us a result (using CLAHE image - _Contrast Limited Adaptive Histogram Equalization_) which is shape of detected face.