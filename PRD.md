# Product Requirements Document (PRD)

## Product Name
**Kobe App** (working title)

---

## Purpose
To help basketball players (and enthusiasts) track, analyze, and improve their shooting form and accuracy using real-time computer vision on their mobile device. The app provides instant feedback, visual overlays, and gamified features to make solo or group practice more engaging and effective.

---

## Target Users
- Basketball players of all skill levels
- Coaches and trainers
- Casual users who want to play and track shots at home (e.g., mini hoop, driveway, gym)
- Anyone interested in computer vision sports tech

---

## Key Features

### 1. Real-Time Camera-Based Detection
- Uses the device camera to detect:
  - The user's pose (skeleton/limbs)
  - The basketball (or any object in hand)
  - The hoop/rim (via model or user-defined area)
- Overlays bounding boxes, skeleton, and other visual cues on the live camera feed.

### 2. User-Defined Hoop Area
- Users can drag to draw a box around the hoop/rim for custom environments.
- The box remains fixed until redefined.
- Clear instructions and feedback during the definition process.

### 3. Pose Estimation & Skeleton Overlay
- Draws keypoints and limb connections (skeleton) over the user in real time.
- Provides visual feedback on form and movement.

### 4. Ball/Object Tracking & Trajectory Visualization
- Detects and tracks the ball (or object in hand) as it moves.
- Draws a blue box around the ball and a cyan line for its recent trajectory.
- (Optional/future) Convex hull visualization for advanced analysis.

### 5. Shot & Make Detection
- Analyzes the ball's trajectory to detect shot attempts and makes:
  - Recognizes an upward and downward arc.
  - Detects if the ball passes through the hoop area and falls below it (or bounces).
- Triggers visual and/or audio feedback for made shots.

### 6. Responsiveness & Robustness
- Handles camera switching, movement, and lighting changes gracefully.
- Keeps overlays and hoop area consistent as the camera moves.

### 7. Gamification & Engagement (Planned)
- Tracks shot count, makes, and streaks.
- Provides instant feedback and celebratory effects for made shots.
- (Future) Leaderboards, challenges, and social sharing.

### 8. Accessibility & Customization (Planned)
- Voice feedback for key events.
- Colorblind-friendly overlays.
- Customizable overlay colors and sounds.

### 9. Onboarding & Tutorial (Planned)
- First-time user tutorial with interactive overlays and instructions.

---

## User Flows

### A. First-Time Setup
1. User opens the app and is guided through a quick tutorial.
2. User is prompted to define the hoop area by dragging a box around the rim.
3. User sees their pose and the hoop area overlaid on the camera feed.

### B. Shooting Session
1. User takes shots; the app tracks the ball and their pose.
2. The app detects shot attempts and makes, providing instant feedback.
3. User can view their shot count, makes, and trajectory overlays.

### C. Redefining Hoop Area
1. User taps a button to redefine the hoop area.
2. User drags a new box; confirms or cancels as needed.

### D. Reviewing Performance (Planned)
1. User can view stats, replays, or highlights of their best shots.

---

## Technical Requirements

- iOS app (Swift/SwiftUI)
- Uses CoreML/Vision for pose and object detection
- Integrates with Roboflow or custom models for basketball/rim detection
- Real-time overlays using SwiftUI
- Efficient, low-latency processing for smooth user experience

---

## Stretch Goals / Future Features

- Multi-person support (track multiple players)
- Advanced shot type classification (layup, jump shot, etc.)
- Replay/highlight system
- Social features (sharing, leaderboards)
- AR enhancements (virtual hoops, effects)
- Android version

---

## What's Working Well
- Real-time detection and overlays
- User-defined hoop area
- Pose skeleton and ball tracking
- Responsive UI and feedback

## What Needs Improvement / In Progress
- More robust shot/make detection (trajectory-based)
- Onboarding/tutorial
- Gamification and engagement features
- Accessibility and customization

---

## Open Questions
- Should the app support other sports or training modes?
- What level of analytics/statistics do users want?
- How much manual vs. automatic setup is ideal for users? 