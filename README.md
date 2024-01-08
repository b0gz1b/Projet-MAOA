# Projet-MAOA : Autour du tracé d'un métro circulaire

This repository contains the code for the project of the course MAOA (Modèles et Applications en Ordonnancement et optimisation combinAtoire)

## Table of contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)

## Introduction

We have a set of points on a map corresponding to the centres of densely populated areas where stations for this line could potentially be installed. We want to draw the route of the line passing through only p of these points: users from the surrounding areas will walk to the nearest station. The aim is to minimise the length of journeys on foot and by public transport for users of the line.

## Requirements

For the heuristic part:

- Python 3.10 or higher
- NumPy library
- Matplotlib library
- scikit-learn library
- networkx library
- PyConcorde library (<https://github.com/jvkersch/pyconcorde>)

To install the libraries, run the following commands:

```bash
pip install numpy matplotlib scikit-learn networkx
pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'
```

For the MIP part you'll need julia and the following packages:

- JuMP
- CPLEX
- Shuffle
- Combinatorics

## Usage

You can run every file separately, code used for the report is written in the files mentioned in the report. Heuristic methods are in the `src` folder and MIP methods are in the `ringStar` folder.
