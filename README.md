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

An example of interesting landmarks:

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

We consider an array between 0 to 1 with 0.1 step with 0.5 valued input. It provides an information it's `average` (between `poor` or `good` values described by auto-membership function).

Defining `rule` of `fear` emotion (skiping `mouth` definition) there's possibility to introduce _Fuzzy Control System_ named `expression_ctrl` which has `expression` - _Simulation_.

It could be computed but it has to be filled by specific values provided by _landmarks_ coordinates and calculations between them.
