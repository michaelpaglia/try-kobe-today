# Kobe App

A real-time basketball training application that uses computer vision to analyze shooting form and provide instant feedback.

## Overview

Kobe App uses your device's camera to track your basketball shooting form in real-time. It combines pose estimation with object detection to analyze shooting mechanics, detect shots, and provide feedback to help improve your game.

## Features

### ✅ Currently Working
- **Real-time pose estimation** - Tracks body skeleton and joint positions using Vision framework
- **Basketball and rim detection** - Uses CoreML and Roboflow API for object detection  
- **Shot detection** - Analyzes ball trajectory and rim interaction to detect successful shots
- **Visual overlays** - Real-time skeleton, bounding boxes, and trajectory visualization
- **Shooting form analysis** - Calculates joint angles (elbow, knee) and provides audio feedback
- **User-defined hoop area** - Manual hoop area definition for custom environments
- **Front/back camera support** - Toggle between camera orientations

### 🔧 Recently Implemented (This Commit)
- **Advanced pose analysis** with joint angle calculations and feedback
- **Enhanced shot detection logic** using trajectory analysis and hoop area detection
- **Audio feedback system** for shooting form correction
- **Improved UI overlays** with confidence-based joint coloring and labels
- **Ball trajectory tracking** with visual path rendering
- **CocoaPods integration** for Roboflow framework
- **CoreML model integration** (best.mlpackage) for local basketball detection
- **Comprehensive pose overlay system** with 17+ joint tracking

## Technical Architecture

### Core Components
- **BasketballTrainerView.swift** - Main training interface with shot analysis
- **CameraView.swift** - Camera capture and ML processing pipeline  
- **PoseOverlayView.swift** - Real-time visual overlays and pose rendering

### ML Pipeline
1. **Vision Framework** - Human pose estimation (`VNHumanBodyPoseObservation`)
2. **CoreML Model** - Local basketball/rim detection (best.mlpackage)
3. **Roboflow API** - Cloud-based object detection service
4. **Shot Analysis** - Custom trajectory and form analysis algorithms

### Dependencies
- **iOS 15.4+** target
- **CocoaPods** - Roboflow framework integration
- **Vision** - Pose estimation
- **CoreML** - Local ML inference
- **AVFoundation** - Camera capture and audio feedback

## Development Setup

```bash
# Install dependencies
pod install

# Open workspace (not project)
open kobe-app.xcworkspace

# Build and run on device (camera required)
```

## Current Implementation Status

### Working Well ✅
- Real-time detection and overlays perform smoothly
- Pose skeleton rendering with confidence visualization
- User-defined hoop area with drag gesture interface
- Shot counting and basic trajectory analysis
- Audio feedback for shooting form

### In Progress 🔧
- **Shot accuracy analysis** - Need trajectory-based make/miss detection
- **Form scoring system** - Quantitative shooting form ratings
- **Session statistics** - Historical tracking and analytics

### Planned Features 📋
- **Onboarding tutorial** - First-time user guidance
- **Advanced analytics** - Shot charts, form progression tracking
- **Social features** - Leaderboards and sharing
- **Multiple player support** - Track multiple people simultaneously
- **Shot type classification** - Detect layups vs jump shots vs free throws

## API Configuration

### Roboflow Integration
- **Model**: "basketball-hoop-images" version 2
- **API Key**: Currently hardcoded in CameraView.swift:295 (needs externalization)
- **Endpoint**: https://detect.roboflow.com/

## Performance Notes

- **Frame Processing**: Limited to every 5th frame for efficiency
- **Detection Intervals**: Minimum 1/30s between ML processing cycles  
- **Coordinate Systems**: Vision framework uses normalized coordinates (0-1)
- **Memory Management**: Trajectory history with configurable size limits

## Known Issues

- API key should be externalized to environment variables
- Shot detection logic needs refinement for edge cases
- Camera orientation changes require app restart
- Roboflow API calls may experience latency

## Contributing

This is currently a personal project. The codebase follows Swift/SwiftUI conventions with extensive documentation in CLAUDE.md.

---

**Last Updated**: June 23, 2025  
**Version**: Initial implementation with advanced pose analysis  
**Platform**: iOS 15.4+