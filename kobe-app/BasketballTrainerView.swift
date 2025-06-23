import SwiftUI
import AVFoundation
import Vision

struct BasketballTrainerView: View {
    @State private var posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
    @State private var poseObservation: VNHumanBodyPoseObservation? = nil
    @State private var shotCount: Int = 0
    @State private var showShotMessage: Bool = false
    @State private var lastWristY: CGFloat? = nil
    @State private var lastElbowY: CGFloat? = nil
    @State private var lastShoulderY: CGFloat? = nil
    @State private var armExtended: Bool = false
    @State private var shotState: ShotState = .waitingForLoad
    @State private var useFrontCamera: Bool = true
    @State private var segmentationMask: CGImage? = nil
    @State private var currentAngle: CGFloat? = nil
    @State private var isInShootingMotion: Bool = false
    @State private var hoopBoundingBoxes: [Detection] = []
    @State private var userDefinedHoopArea: CGRect? = nil
    @State private var shotDetected: Bool = false
    @State private var personLocation: CGRect? = nil
    @State private var isDefiningHoop: Bool = false
    @State private var hoopStartPoint: CGPoint? = nil
    @State private var hoopEndPoint: CGPoint? = nil
    @State private var dragStartPoint: CGPoint? = nil
    @State private var dragEndPoint: CGPoint? = nil
    @State private var ballTrajectory: [CGPoint] = []
    @State private var ballBox: CGRect? = nil
    
    // Indices for right arm keypoints in Vision's output (approximate order)
    let rightWristIndex = 14
    let rightElbowIndex = 13
    let rightShoulderIndex = 12
    let rightKneeIndex = 26
    let rightAnkleIndex = 28
    let leftKneeIndex = 25
    let leftAnkleIndex = 27
    let leftHipIndex = 23
    let rightHipIndex = 24
    
    let elbowGoodRange: ClosedRange<CGFloat> = 160...180
    let kneeGoodRange: ClosedRange<CGFloat> = 150...180 // nearly straight at release
    let minFeetDistance: CGFloat = 80 // pixels, adjust as needed
    let speechSynth = AVSpeechSynthesizer()
    
    enum ShotState {
        case waitingForLoad, loaded, extended
    }
    
    var body: some View {
        ZStack {
            CameraView(
                posePoints: $posePoints,
                poseObservation: $poseObservation,
                useFrontCamera: useFrontCamera,
                segmentationMask: $segmentationMask,
                hoopBoundingBoxes: $hoopBoundingBoxes,
                userDefinedHoopArea: $userDefinedHoopArea,
                shotDetected: $shotDetected,
                personLocation: $personLocation
            )
            .edgesIgnoringSafeArea(.all)
            
            PoseOverlayView(
                points: posePoints,
                observation: poseObservation,
                currentAngle: currentAngle,
                hoopBoundingBoxes: hoopBoundingBoxes,
                personLocation: personLocation,
                shotDetected: shotDetected,
                userDefinedHoopArea: userDefinedHoopArea,
                ballTrajectory: ballTrajectory,
                ballBox: ballBox,
                useFrontCamera: useFrontCamera
            )
            .allowsHitTesting(false)
            
            VStack {
                HStack {
                    Spacer()
                    Button(action: { useFrontCamera.toggle() }) {
                        Image(systemName: useFrontCamera ? "camera.rotate" : "camera")
                            .font(.title)
                            .padding(10)
                            .background(Color.black.opacity(0.5))
                            .foregroundColor(.white)
                            .clipShape(Circle())
                    }
                    .padding(.top, 40)
                    .padding(.trailing, 20)
                }
                
                Spacer()
                
                // Hoop definition controls (temporarily disabled)
                // if !isDefiningHoop {
                //     Button(action: { isDefiningHoop = true }) {
                //         Text("Define Hoop Area")
                //             .font(.headline)
                //             .padding()
                //             .background(Color.blue)
                //             .foregroundColor(.white)
                //             .cornerRadius(10)
                //     }
                //     .padding(.bottom, 20)
                // }
                
                Text("Basketball Trainer")
                    .font(.title)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                
                Text("Shots: \(shotCount)")
                    .font(.headline)
                    .padding(.horizontal)
                    .background(Color.black.opacity(0.5))
                    .foregroundColor(.white)
                    .cornerRadius(8)
                
                if showShotMessage {
                    Text("Shot detected!")
                        .font(.headline)
                        .foregroundColor(.green)
                        .padding(.horizontal)
                        .background(Color.black.opacity(0.7))
                        .cornerRadius(8)
                }
            }
            .padding(.bottom, 40)
            
            // Hoop definition overlay (temporarily disabled)
            if false { // isDefiningHoop {
                GeometryReader { geometry in
                    ZStack {
                        Color.black.opacity(0.3)
                            .edgesIgnoringSafeArea(.all)
                            .gesture(
                                DragGesture(minimumDistance: 0)
                                    .onChanged { value in
                                        if dragStartPoint == nil {
                                            dragStartPoint = value.startLocation
                                        }
                                        dragEndPoint = value.location
                                    }
                                    .onEnded { value in
                                        dragEndPoint = value.location
                                    }
                            )

                        // Draw the live-updating rectangle as the user drags
                        if let start = dragStartPoint, let end = dragEndPoint {
                            let rect = CGRect(
                                x: min(start.x, end.x),
                                y: min(start.y, end.y),
                                width: abs(end.x - start.x),
                                height: abs(end.y - start.y)
                            )
                            Rectangle()
                                .stroke(Color.yellow, lineWidth: 2)
                                .background(Color.yellow.opacity(0.2))
                                .frame(width: rect.width, height: rect.height)
                                .position(x: rect.midX, y: rect.midY)

                            // Confirm and Cancel buttons
                            VStack {
                                Spacer()
                                HStack {
                                    Button(action: {
                                        // Cancel
                                        dragStartPoint = nil
                                        dragEndPoint = nil
                                    }) {
                                        Text("Cancel")
                                            .font(.headline)
                                            .padding()
                                            .background(Color.red)
                                            .foregroundColor(.white)
                                            .cornerRadius(10)
                                    }
                                    .padding(.leading, 30)
                                    Spacer()
                                    Button(action: {
                                        // Confirm
                                        let startX = rect.minX / geometry.size.width
                                        let startY = rect.minY / geometry.size.height
                                        let width = rect.width / geometry.size.width
                                        let height = rect.height / geometry.size.height
                                        userDefinedHoopArea = CGRect(x: startX, y: startY, width: width, height: height)
                                        isDefiningHoop = false
                                        dragStartPoint = nil
                                        dragEndPoint = nil
                                    }) {
                                        Text("Confirm")
                                            .font(.headline)
                                            .padding()
                                            .background(Color.green)
                                            .foregroundColor(.white)
                                            .cornerRadius(10)
                                    }
                                    .padding(.trailing, 30)
                                }
                                .padding(.bottom, 40)
                            }
                        }

                        Text("Drag to draw hoop area")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(10)
                            .padding(.top, 60)
                    }
                }
            }
        }
        .onChange(of: shotDetected) { oldValue, newValue in
            if newValue {
                shotCount += 1
                showShotMessage = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    showShotMessage = false
                }
            }
        }
    }
    
    func getPoint(_ name: VNHumanBodyPoseObservation.JointName, from recognizedPoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint], minConfidence: Float = 0.3) -> CGPoint? {
        if let point = recognizedPoints[name], point.confidence > minConfidence {
            return CGPoint(x: point.x, y: 1 - point.y)
        }
        return nil
    }
    
    // Helper: Returns true if the right knee is significantly higher than the left (for right-handed layup)
    func isRightKneeDrive(points: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]) -> Bool {
        guard let rightKnee = points[.rightKnee], let leftKnee = points[.leftKnee], rightKnee.confidence > 0.3, leftKnee.confidence > 0.3 else { return false }
        return rightKnee.y < leftKnee.y - 0.05 // adjust threshold for normalized coordinates
    }
    
    // Helper: Returns true if both feet are off the ground (for jump shot)
    func areBothFeetOffGround(points: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]) -> Bool {
        guard let leftAnkle = points[.leftAnkle], let rightAnkle = points[.rightAnkle], leftAnkle.confidence > 0.3, rightAnkle.confidence > 0.3 else { return false }
        let groundThreshold: CGFloat = 0.1 // normalized coordinate, adjust as needed
        return leftAnkle.y < groundThreshold && rightAnkle.y < groundThreshold
    }
    
    // Main function to classify and give feedback
    func analyzeShotType(points: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]) {
        let layup = isRightKneeDrive(points: points)
        let jumpShot = areBothFeetOffGround(points: points)
        
        if layup {
            speakFeedback("Great layup form! Drive your knee up and finish strong.")
        } else if jumpShot {
            speakFeedback("Good jump shot! Keep your balance and follow through.")
        } else {
            speakFeedback("Adjust your form for a better shot.")
        }
    }
    
    func detectShot(from recognizedPoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]) {
        let rightWrist = getPoint(.rightWrist, from: recognizedPoints)
        let rightElbow = getPoint(.rightElbow, from: recognizedPoints)
        let rightShoulder = getPoint(.rightShoulder, from: recognizedPoints)
        let rightHip = getPoint(.rightHip, from: recognizedPoints)
        let rightKnee = getPoint(.rightKnee, from: recognizedPoints)
        let rightAnkle = getPoint(.rightAnkle, from: recognizedPoints)
        let leftHip = getPoint(.leftHip, from: recognizedPoints)
        let leftKnee = getPoint(.leftKnee, from: recognizedPoints)
        let leftAnkle = getPoint(.leftAnkle, from: recognizedPoints)

        guard let wrist = rightWrist, let elbow = rightElbow, let shoulder = rightShoulder,
              let rHip = rightHip, let rKnee = rightKnee, let rAnkle = rightAnkle,
              let lHip = leftHip, let lKnee = leftKnee, let lAnkle = leftAnkle else {
            print("Missing keypoints, skipping frame")
            return
        }

        // Calculate current angle
        if let angle = angleBetween(jointA: shoulder, jointB: elbow, jointC: wrist) {
            currentAngle = angle
        }

        let wristY = wrist.y
        let elbowY = elbow.y
        let shoulderY = shoulder.y
        let leftFoot = lAnkle
        let rightFoot = rAnkle
        let elbowAngle = angleBetween(jointA: shoulder, jointB: elbow, jointC: wrist) ?? 0
        let rightKneeAngle = angleBetween(jointA: rHip, jointB: rKnee, jointC: rAnkle) ?? 0
        let leftKneeAngle = angleBetween(jointA: lHip, jointB: lKnee, jointC: lAnkle) ?? 0
        let feetDistance = hypot(leftFoot.x - rightFoot.x, leftFoot.y - rightFoot.y)

        // Detect shooting motion
        let isWristAboveShoulder = wristY < shoulderY - 0.1
        let isElbowAboveShoulder = elbowY < shoulderY - 0.05
        let isArmExtended = elbowAngle > 150

        switch shotState {
        case .waitingForLoad:
            if isWristAboveShoulder && isElbowAboveShoulder {
                shotState = .loaded
                isInShootingMotion = true
            }
        case .loaded:
            if isArmExtended && isWristAboveShoulder {
                shotState = .extended
            }
            if wristY > shoulderY + 0.1 {
                // Shot release detected
                shotCount += 1
                showShotMessage = true
                var feedbacks: [String] = []
                
                // Analyze form
                if !elbowGoodRange.contains(elbowAngle) {
                    feedbacks.append("Straighten your elbow!")
                }
                if !kneeGoodRange.contains(rightKneeAngle) || !kneeGoodRange.contains(leftKneeAngle) {
                    feedbacks.append("Straighten your knees!")
                }
                if feetDistance < minFeetDistance {
                    feedbacks.append("Widen your stance!")
                }
                
                // Provide angle feedback
                if let angle = currentAngle {
                    if angle < 80 {
                        feedbacks.append("Release point too low!")
                    } else if angle > 100 {
                        feedbacks.append("Release point too high!")
                    }
                }
                
                // Layup vs. jump shot feedback
                analyzeShotType(points: recognizedPoints)
                
                if feedbacks.isEmpty {
                    speakFeedback("Good form!")
                } else {
                    for feedback in feedbacks {
                        speakFeedback(feedback)
                    }
                }
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    showShotMessage = false
                }
                shotState = .waitingForLoad
                isInShootingMotion = false
            }
        case .extended:
            if wristY > shoulderY + 0.1 {
                shotState = .waitingForLoad
                isInShootingMotion = false
            }
        }
        
        lastWristY = wristY
        lastElbowY = elbowY
        lastShoulderY = shoulderY
    }
    
    func angleBetween(jointA: CGPoint?, jointB: CGPoint?, jointC: CGPoint?) -> CGFloat? {
        guard let a = jointA, let b = jointB, let c = jointC else { return nil }
        let ab = CGVector(dx: a.x - b.x, dy: a.y - b.y)
        let cb = CGVector(dx: c.x - b.x, dy: c.y - b.y)
        let dotProduct = ab.dx * cb.dx + ab.dy * cb.dy
        let magnitudeAB = sqrt(ab.dx * ab.dx + ab.dy * ab.dy)
        let magnitudeCB = sqrt(cb.dx * cb.dx + cb.dy * cb.dy)
        guard magnitudeAB > 0, magnitudeCB > 0 else { return nil }
        let angle = acos(dotProduct / (magnitudeAB * magnitudeCB))
        return angle * 180 / .pi
    }
    
    func speakFeedback(_ message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        speechSynth.speak(utterance)
    }
}

#Preview {
    BasketballTrainerView()
} 