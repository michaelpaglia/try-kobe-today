# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kobe App is an iOS basketball training application that uses real-time computer vision to track shooting form, detect shots, and provide instant feedback. The app combines pose estimation with object detection to analyze basketball shooting mechanics and gamify practice sessions.

## Architecture

### Core Components

- **BasketballTrainerView.swift**: Main training interface that orchestrates the camera, pose detection, and shot analysis
- **CameraView.swift**: UIViewControllerRepresentable wrapper for CameraViewController with extensive ML integration
- **PoseOverlayView.swift**: SwiftUI overlay that renders pose skeletons, bounding boxes, and trajectory visualizations
- **ContentView.swift**: Simple navigation entry point to the trainer

### ML Integration

The app uses multiple ML frameworks:
- **Vision Framework**: Human pose estimation using `VNHumanBodyPoseObservation` for skeletal tracking
- **CoreML**: Custom YOLO model for basketball/rim detection (best.mlpackage)
- **Roboflow**: Third-party basketball detection service with API integration

### Detection Pipeline

1. Camera captures frames at 30 FPS target
2. Vision framework processes pose estimation in parallel
3. Roboflow API processes object detection (basketball, rim, person)
4. Shot detection logic analyzes ball trajectory relative to hoop area
5. Kalman filtering smooths tracking data
6. UI overlays render real-time feedback

## Development Commands

### Building and Running
```bash
# Open workspace (includes CocoaPods)
open kobe-app.xcworkspace

# Install/update pods
pod install
```

### Testing
The project includes standard Xcode test targets:
- `kobe-appTests`: Unit tests
- `kobe-appUITests`: UI automation tests

Run tests through Xcode Test Navigator or:
```bash
xcodebuild test -workspace kobe-app.xcworkspace -scheme kobe-app -destination 'platform=iOS Simulator,name=iPhone 15'
```

## Key Configuration

### CocoaPods Dependencies
- **Roboflow**: Computer vision API for basketball/hoop detection
- Target iOS 15.4+

### API Keys
- Roboflow API key is hardcoded in CameraView.swift:295 (should be externalized)
- Model: "basketball-hoop-images" version 2

### ML Models
- **best.mlpackage**: Custom CoreML model for basketball detection
- **yolov8n.pt**: PyTorch model file (unused in current implementation)

## Core Features Implementation

### Shot Detection Logic
Located in `CameraViewController.detectShot()`:
- Tracks ball leaving person's vicinity
- Detects ball entering rim bounding box
- Requires 3+ consecutive frames in hoop for shot confirmation
- Resets detection state after 1 second

### Pose Analysis
Located in `BasketballTrainerView.detectShot()`:
- Analyzes shooting form using joint angles
- Provides audio feedback for form correction
- Tracks shooting state machine: waitingForLoad → loaded → extended
- Calculates elbow, knee angles and stance width

### User-Defined Hoop Area
- Drag gesture interface for manual hoop area definition
- Stored as normalized CGRect coordinates
- Overrides automatic rim detection when set

## Development Notes

### Performance Optimizations
- Frame processing limited to every 5th frame
- Minimum 1/30s interval between processing cycles
- Motion detection to clear stale detections
- Kalman filtering for smooth object tracking

### Coordinate Systems
- Vision framework uses normalized coordinates (0-1)
- UI coordinates converted for SwiftUI overlays
- Roboflow detection results normalized to frame dimensions

### State Management
- Extensive use of @State for real-time UI updates
- Detection callbacks from UIKit to SwiftUI via closures
- Trajectory history maintained with configurable size limits