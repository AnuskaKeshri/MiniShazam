# EE200 Mini Shazam - Audio Fingerprinting System

## Project Overview

This project implements a Shazam-style audio recognition system using audio fingerprinting.

The system extracts important features from songs, creates fingerprints, stores them in a database, and recognizes an unknown query audio clip.

Developed for:
EE200 Signals to Software Project


## Features

- Audio fingerprint generation
- Song database indexing
- Query audio recognition
- Spectrogram visualization
- Constellation map visualization
- Offset histogram matching
- Single clip recognition mode
- Batch recognition mode
- CSV result generation


## Application Modes


### 1. Identify Song

Upload a single audio clip.

The application displays:

- Recognized song name
- Number of matching votes
- Spectrogram
- Constellation map
- Offset histogram


### 2. Batch Mode

Upload multiple audio clips.

The application generates:

filename,prediction

CSV format exactly as required.
