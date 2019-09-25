# Face-Expression-Recognition

Face expression recognition using fuzzy control system (scikit-fuzzy)

## Requirements

* _Jupyter Notebook_ with _Python 3.7_
* Libs:

- numpy
- opencv-python
- scikit-fuzzy
- Pillow
- dlib (*Caution:* _Anaconda_'s `conda install -c conda-forge dlib` would be useful)
- mss (_ver. 2.0.22_)


## Introduction

An app is designed to detect and recognize face expression (and partially expressions). It uses fuzzy logic to determine which emotion is portraited on the picture.

## Motivation

#### Amanda Patterson

Body-language's sheets described by _Amanda Patterson_ provides an information about specific "translations". It's best way to understand the purposes of the experiment and idea of image recognition shown below.

_Example:_

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/readme-images/Cheat-Sheets-For-Body-Language-Part-1.jpg)

#### Paul Ekman

Main character of tv-series _Lie to Me_ (played by an american actor _Tim Roth_) is psychologist being an extraoridinary specialist of emotions recognition based on facial expressions.

_Example:_

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

_An example of interesting landmarks:_

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/readme-images/example-of-interesting-landmarks.PNG)

#### Skfuzzy Control System

Going through an eyebrows example:

```python
eyebrows = ctrl.Antecedent(np.arange(0, 1, 0.1), 'eyebrows')
eyebrows.automf(3)

rule = ctrl.Rule(eyebrows['poor'] & mouth['good'], emotion['fear'])

expression_ctrl = ctrl.ControlSystem([rule])
expression = ctrl.ControlSystemSimulation(expression_ctrl)

expression.input['eyebrows'] = 0.5
expression.compute()
```

We consider an array between _0_ to _1_ with _0.1_ step with _0.5_ valued input. It provides an information it's `average` (between `poor` or `good` values described by _auto-membership function_).

Defining `rule` of `fear` emotion (skiping `mouth` definition) there's possibility to introduce _Fuzzy Control System_ named `expression_ctrl` which has `expression` - _Simulation_.

It could be computed but it has to be filled by specific values provided by _landmarks_ coordinates and calculations between them.

#### Dummy Eyebrows-Expression's Input

Going through specific coordinates which are defined by `Shape Predictor` authors:

```python
# Eyebrows down
if shape.part(leftPartOfLeftEyebrow).y < shape.part(rightPartOfLeftEyebrow).y or shape.part(leftPartOfRightEyebrow).y < shape.part(rightPartOfRightEyebrow).y:
	expression.input['eyebrows'] = 0
else:
	expression.input['eyebrows'] = 0.5
```	

_Assumption:_

The center of the eyebrows is lower than those side-borders. It provides us an `eyebrows` value (_Skfuzzy Control System_ input) as `poor`. It could means that the person is in anger as shown below.

_Example:_

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/readme-images/eyebrows-down.PNG)

It's simplest example but values could be also more sophisticated (using _landmarks_ like above there could be geometrical calculation like angle or more specific form that provides more than 3 states of `eyebrows`).

Although _Smile Detection_ presented in source-code is valued between 0 to 1 (smiled or not) there is shown interesting method called _Haar Cascades Detection_ which allows to use ready-made and well trained _xml-cascade_-s using `OpenCV`'s `CascadeClassifier`.

#### Result

Application uses `mss` library to capture an image from selected on-screen location.* Captured image is processed by following code after _Q_-keypressed.

_Example_:

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/readme-images/anger.PNG)

```python
emotion.view(sim=expression)
```

As a result of our specified "emotions universe" with `expression` _simulation_ there would be viewed a diagram of fuzzed values provided in mentioned rules and definitions based on _Tim Roth Coefficient_, as below:

![](https://github.com/khoczkiewicz/Face-Expression-Recognition/blob/master/readme-images/trc-anger.PNG)

`*` - (The reason is I've got no camera on my computer, so I do use _Playstation 3_ camera, which has no proper drivers on _Windows 10_).