# C-PIA


## Introduction
This repository hosts the Python code for the Causal - Piano Inpainting Application (C-PIA), which is a Max4Live device performing *pianoroll inpainting* in Ableton.
C-PIA is an improvement of our previous work PIA (https://ghadjeres.github.io/piano-inpainting-application/).
Both plug-in allow a user to select a region of a score which he or she wants to modify, and the algorithm makes a new proposal for that region.
The algorithms are trained using machine learning, specifically linear transformers.
PIA was constrained to generate a fixed number of notes in the inpainted region.
By modifying the conditioning information in order to have a purely causal model, C-PIA is freed from this constraint.

## Installation
Please use the requirement.txt file to setup your python environment since some libraries require specific versions.

    pip install -r requirements.txt

## Download pretrained models

## Inpainting

## Training your own model

