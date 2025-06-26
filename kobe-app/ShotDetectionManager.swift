import Foundation
import Vision
import CoreGraphics
import QuartzCore

// MARK: - Shot Detection States
enum ShotPhase: String, CaseIterable {
    case ready = "Ready"           // Person holding ball, in shooting stance
    case preparation = "Loading"   // Bringing ball up, loading motion
    case release = "Release"       // Ball leaving shooting hand
    case followThrough = "Follow"  // Arm extended after release
    case complete = "Complete"     // Shot completed
}

enum ShootingHand {
    case left
    case right
    case unknown
}

// MARK: - Shot Detection Result
struct ShotDetectionResult {
    let phase: ShotPhase
    let shootingHand: ShootingHand
    let confidence: Float
    let ballPosition: CGPoint?
    let releasePosition: CGPoint?
    let timestamp: TimeInterval
    let shootingForm: ShootingFormAnalysis?
}

// MARK: - Shooting Form Analysis
struct ShootingFormAnalysis {
    let elbowAngle: CGFloat?        // Shooting elbow angle at release
    let shoulderAlignment: CGFloat?  // Shoulder square to target
    let followThroughAngle: CGFloat? // Wrist snap angle
    let footStance: CGFloat?        // Foot positioning
    let bodyBalance: CGFloat?       // Center of gravity
    let arcQuality: Float          // 0-1 score for shot arc
}

// MARK: - Advanced Shot Detection Manager
class ShotDetectionManager {
    
    // MARK: - Configuration
    private struct Config {
        static let handProximityThreshold: CGFloat = 0.12      // Ball near hand distance
        static let releaseVelocityThreshold: CGFloat = 0.08    // Higher velocity for actual release
        static let elbowExtensionThreshold: CGFloat = 160.0    // Much higher - must be nearly straight
        static let phaseTimeoutSeconds: TimeInterval = 5.0     // Longer timeout for real shots
        static let minimumShotDuration: TimeInterval = 0.8     // Longer minimum for full shot
        static let followThroughDuration: TimeInterval = 1.5   // Longer follow-through requirement
        static let ballLeftHandThreshold: CGFloat = 0.25       // Ball must move far from hand
        static let armDownThreshold: CGFloat = 140.0           // Arm angle when returning down
    }
    
    // MARK: - Private Properties
    private var currentPhase: ShotPhase = .ready
    private var phaseStartTime: TimeInterval = 0
    private var shotStartTime: TimeInterval = 0
    private var lastBallPosition: CGPoint?
    private var lastPosePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
    private var shootingHand: ShootingHand = .unknown
    private var releasePosition: CGPoint?
    private var shotHistory: [ShotDetectionResult] = []
    
    // Hand position tracking for release detection
    private var leftHandHistory: [CGPoint] = []
    private var rightHandHistory: [CGPoint] = []
    private let handHistorySize = 5
    
    // MARK: - Public Interface
    
    /// Process current frame for shot detection
    func processShotDetection(
        ballPosition: CGPoint?,
        ballVelocity: CGVector,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool = false,
        currentTime: TimeInterval = CACurrentMediaTime()
    ) -> ShotDetectionResult? {
        
        // Store current data
        lastBallPosition = ballPosition
        lastPosePoints = posePoints
        
        // Determine shooting hand if unknown
        if shootingHand == .unknown {
            shootingHand = determineShootingHand(posePoints: posePoints, ballPosition: ballPosition, useFrontCamera: useFrontCamera)
        }
        
        // Process state machine
        let newPhase = determineNextPhase(
            currentPhase: currentPhase,
            ballPosition: ballPosition,
            ballVelocity: ballVelocity,
            posePoints: posePoints,
            useFrontCamera: useFrontCamera,
            currentTime: currentTime
        )
        
        // Handle phase transitions
        if newPhase != currentPhase {
            handlePhaseTransition(from: currentPhase, to: newPhase, currentTime: currentTime)
            currentPhase = newPhase
            phaseStartTime = currentTime
        }
        
        // Check for phase timeout
        if currentTime - phaseStartTime > Config.phaseTimeoutSeconds && currentPhase != .ready {
            print("[ShotDetection] ⏰ Phase timeout, resetting to ready")
            resetShotDetection()
        }
        
        // Analyze shooting form
        let formAnalysis = analyzeShootingForm(
            posePoints: posePoints,
            ballPosition: ballPosition,
            phase: currentPhase,
            useFrontCamera: useFrontCamera
        )
        
        // Create result
        let result = ShotDetectionResult(
            phase: currentPhase,
            shootingHand: shootingHand,
            confidence: calculateConfidence(phase: currentPhase, posePoints: posePoints),
            ballPosition: ballPosition,
            releasePosition: releasePosition,
            timestamp: currentTime,
            shootingForm: formAnalysis
        )
        
        return result
    }
    
    /// Reset shot detection state
    func resetShotDetection() {
        currentPhase = .ready
        phaseStartTime = CACurrentMediaTime()
        shotStartTime = 0
        releasePosition = nil
        shootingHand = .unknown
        leftHandHistory.removeAll()
        rightHandHistory.removeAll()
    }
    
    // MARK: - Private Methods
    
    private func determineShootingHand(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        ballPosition: CGPoint?,
        useFrontCamera: Bool
    ) -> ShootingHand {
        
        guard let ballPosition = ballPosition else { return .unknown }
        
        // Get hand positions
        guard let leftWrist = posePoints[.leftWrist],
              let rightWrist = posePoints[.rightWrist],
              leftWrist.confidence > 0.5, rightWrist.confidence > 0.5 else {
            return .unknown
        }
        
        // Convert coordinates (handle front camera mirroring)
        let leftX = useFrontCamera ? (1.0 - leftWrist.x) : leftWrist.x
        let rightX = useFrontCamera ? (1.0 - rightWrist.x) : rightWrist.x
        let leftPos = CGPoint(x: leftX, y: 1.0 - leftWrist.y)
        let rightPos = CGPoint(x: rightX, y: 1.0 - rightWrist.y)
        
        // Calculate distances to ball
        let leftDistance = distance(from: leftPos, to: ballPosition)
        let rightDistance = distance(from: rightPos, to: ballPosition)
        
        // Shooting hand is typically the one closer to ball during preparation
        if leftDistance < rightDistance && leftDistance < Config.handProximityThreshold {
            return .left
        } else if rightDistance < leftDistance && rightDistance < Config.handProximityThreshold {
            return .right
        }
        
        return .unknown
    }
    
    private func determineNextPhase(
        currentPhase: ShotPhase,
        ballPosition: CGPoint?,
        ballVelocity: CGVector,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> ShotPhase {
        
        switch currentPhase {
        case .ready:
            return detectPreparationPhase(ballPosition: ballPosition, posePoints: posePoints, useFrontCamera: useFrontCamera)
            
        case .preparation:
            return detectReleasePhase(ballPosition: ballPosition, ballVelocity: ballVelocity, posePoints: posePoints, useFrontCamera: useFrontCamera, currentTime: currentTime)
            
        case .release:
            return detectFollowThroughPhase(posePoints: posePoints, currentTime: currentTime)
            
        case .followThrough:
            return detectCompletionPhase(currentTime: currentTime)
            
        case .complete:
            // Auto-reset after completion
            return .ready
        }
    }
    
    private func detectPreparationPhase(
        ballPosition: CGPoint?,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool
    ) -> ShotPhase {
        
        guard let ballPosition = ballPosition else { return .ready }
        
        // Check if ball is being brought up near shooting hand
        let shootingHandJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        guard let handPoint = posePoints[shootingHandJoint],
              handPoint.confidence > 0.5 else { return .ready }
        
        let handX = useFrontCamera ? (1.0 - handPoint.x) : handPoint.x
        let handPos = CGPoint(x: handX, y: 1.0 - handPoint.y)
        
        let distanceToHand = distance(from: ballPosition, to: handPos)
        
        // Ball is near shooting hand AND hand is elevated (shooting preparation)
        // But arm must NOT be fully extended yet (that would be release)
        if distanceToHand < Config.handProximityThreshold && handPos.y < 0.6 {
            // Check arm angle - should be bent in preparation, not extended
            if let armAngle = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand) {
                if armAngle < 150.0 {  // Arm still bent in preparation
                    print("[ShotDetection] 🏀 PREPARATION PHASE: Ball near \(shootingHand) hand, arm bent (\(String(format: "%.1f", armAngle))°)")
                    return .preparation
                } else {
                    // Arm is extended but ball still near hand - not release yet, stay in prep
                    print("[ShotDetection] 🔄 PREPARATION HOLD: Arm extended (\(String(format: "%.1f", armAngle))°) but ball still near hand")
                    return .preparation
                }
            } else {
                // No arm angle data, use basic proximity
                print("[ShotDetection] 🏀 PREPARATION PHASE: Ball near \(shootingHand) hand")
                return .preparation
            }
        }
        
        return .ready
    }
    
    private func detectReleasePhase(
        ballPosition: CGPoint?,
        ballVelocity: CGVector,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> ShotPhase {
        
        guard let ballPosition = ballPosition else { return .preparation }
        
        // Track hand movement
        updateHandHistory(posePoints: posePoints, useFrontCamera: useFrontCamera)
        
        // Check for ball leaving shooting hand vicinity
        let shootingHandJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        guard let handPoint = posePoints[shootingHandJoint],
              handPoint.confidence > 0.5 else { return .preparation }
        
        let handX = useFrontCamera ? (1.0 - handPoint.x) : handPoint.x
        let handPos = CGPoint(x: handX, y: 1.0 - handPoint.y)
        
        let distanceToHand = distance(from: ballPosition, to: handPos)
        let ballSpeed = sqrt(ballVelocity.dx * ballVelocity.dx + ballVelocity.dy * ballVelocity.dy)
        
        // Release conditions: ball moved FAR from hand AND has STRONG upward velocity
        if distanceToHand > Config.ballLeftHandThreshold && 
           ballSpeed > Config.releaseVelocityThreshold &&
           ballVelocity.dy < -0.03 {  // MUST be strongly upward (negative Y = up)
            
            releasePosition = ballPosition
            print("[ShotDetection] 🚀 RELEASE PHASE: Ball released \(shootingHand) hand, velocity: dx=\(String(format: "%.3f", ballVelocity.dx)), dy=\(String(format: "%.3f", ballVelocity.dy))")
            return .release
        }
        
        // Also check arm extension for release detection - MUST be nearly straight
        if let armExtensionAngle = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand) {
            if armExtensionAngle > Config.elbowExtensionThreshold && distanceToHand > Config.handProximityThreshold * 2 {
                print("[ShotDetection] 🚀 RELEASE PHASE: Full arm extension (\(String(format: "%.1f", armExtensionAngle))°) + ball distance")
                return .release
            }
        }
        
        return .preparation
    }
    
    private func detectFollowThroughPhase(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        currentTime: TimeInterval
    ) -> ShotPhase {
        
        // Check for proper follow-through form (arm extended, wrist snapped)
        let shootingElbow: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftElbow : .rightElbow
        let shootingWrist: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        guard let elbowPoint = posePoints[shootingElbow],
              let wristPoint = posePoints[shootingWrist],
              elbowPoint.confidence > 0.5, wristPoint.confidence > 0.5 else {
            return .release
        }
        
        // Arm should still be extended after release
        if wristPoint.y < elbowPoint.y {  // Wrist higher than elbow = extended arm
            print("[ShotDetection] 📐 FOLLOW-THROUGH PHASE: Arm extended")
            return .followThrough
        }
        
        return .release
    }
    
    private func detectCompletionPhase(currentTime: TimeInterval) -> ShotPhase {
        // Check if arm has returned to non-extended position (shot truly complete)
        let shootingElbow: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftElbow : .rightElbow
        let shootingWrist: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        if let armAngle = calculateArmExtension(posePoints: lastPosePoints, shootingHand: shootingHand) {
            // Shot complete when arm comes back down (less extended)
            if armAngle < Config.armDownThreshold && currentTime - phaseStartTime > 1.0 {
                print("[ShotDetection] ✅ SHOT COMPLETE: Arm returned down (\(String(format: "%.1f", armAngle))°)")
                return .complete
            }
        }
        
        // Fallback: minimum follow-through duration
        if currentTime - phaseStartTime > Config.followThroughDuration {
            print("[ShotDetection] ✅ SHOT COMPLETE: Timeout")
            return .complete
        }
        
        return .followThrough
    }
    
    private func handlePhaseTransition(from oldPhase: ShotPhase, to newPhase: ShotPhase, currentTime: TimeInterval) {
        if oldPhase == .ready && newPhase == .preparation {
            shotStartTime = currentTime
        }
        
        // Reduced logging for performance
    }
    
    // MARK: - Helper Methods
    
    private func distance(from point1: CGPoint, to point2: CGPoint) -> CGFloat {
        return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2))
    }
    
    private func updateHandHistory(posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint], useFrontCamera: Bool) {
        if let leftWrist = posePoints[.leftWrist], leftWrist.confidence > 0.5 {
            let leftX = useFrontCamera ? (1.0 - leftWrist.x) : leftWrist.x
            let leftPos = CGPoint(x: leftX, y: 1.0 - leftWrist.y)
            leftHandHistory.append(leftPos)
            if leftHandHistory.count > handHistorySize {
                leftHandHistory.removeFirst()
            }
        }
        
        if let rightWrist = posePoints[.rightWrist], rightWrist.confidence > 0.5 {
            let rightX = useFrontCamera ? (1.0 - rightWrist.x) : rightWrist.x
            let rightPos = CGPoint(x: rightX, y: 1.0 - rightWrist.y)
            rightHandHistory.append(rightPos)
            if rightHandHistory.count > handHistorySize {
                rightHandHistory.removeFirst()
            }
        }
    }
    
    private func calculateArmExtension(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        shootingHand: ShootingHand
    ) -> CGFloat? {
        
        let shoulder: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftShoulder : .rightShoulder
        let elbow: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftElbow : .rightElbow
        let wrist: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        guard let shoulderPoint = posePoints[shoulder],
              let elbowPoint = posePoints[elbow],
              let wristPoint = posePoints[wrist],
              shoulderPoint.confidence > 0.5,
              elbowPoint.confidence > 0.5,
              wristPoint.confidence > 0.5 else { return nil }
        
        // Calculate angle between shoulder-elbow and elbow-wrist vectors
        let shoulderElbow = CGVector(dx: elbowPoint.x - shoulderPoint.x, dy: elbowPoint.y - shoulderPoint.y)
        let elbowWrist = CGVector(dx: wristPoint.x - elbowPoint.x, dy: wristPoint.y - elbowPoint.y)
        
        let dot = shoulderElbow.dx * elbowWrist.dx + shoulderElbow.dy * elbowWrist.dy
        let mag1 = sqrt(shoulderElbow.dx * shoulderElbow.dx + shoulderElbow.dy * shoulderElbow.dy)
        let mag2 = sqrt(elbowWrist.dx * elbowWrist.dx + elbowWrist.dy * elbowWrist.dy)
        
        guard mag1 > 0, mag2 > 0 else { return nil }
        
        let cosAngle = dot / (mag1 * mag2)
        let angle = acos(max(-1, min(1, cosAngle))) * 180 / .pi
        
        // Return extension angle (180° = fully extended)
        return abs(180 - angle)
    }
    
    private func analyzeShootingForm(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        ballPosition: CGPoint?,
        phase: ShotPhase,
        useFrontCamera: Bool
    ) -> ShootingFormAnalysis? {
        
        // Only analyze form during key phases
        guard phase == .preparation || phase == .release || phase == .followThrough else { return nil }
        
        let elbowAngle = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand)
        
        return ShootingFormAnalysis(
            elbowAngle: elbowAngle,
            shoulderAlignment: nil,  // TODO: Implement
            followThroughAngle: nil, // TODO: Implement  
            footStance: nil,         // TODO: Implement
            bodyBalance: nil,        // TODO: Implement
            arcQuality: 0.0         // TODO: Implement
        )
    }
    
    private func calculateConfidence(
        phase: ShotPhase,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
    ) -> Float {
        
        // Base confidence on pose point quality
        let relevantJoints: [VNHumanBodyPoseObservation.JointName] = [
            .leftShoulder, .rightShoulder, .leftElbow, .rightElbow, .leftWrist, .rightWrist
        ]
        
        let confidences = relevantJoints.compactMap { posePoints[$0]?.confidence }
        guard !confidences.isEmpty else { return 0.0 }
        
        let avgConfidence = confidences.reduce(0, +) / Float(confidences.count)
        
        // Boost confidence for active shooting phases
        switch phase {
        case .ready: return avgConfidence * 0.5
        case .preparation: return avgConfidence * 0.8
        case .release: return avgConfidence * 1.0
        case .followThrough: return avgConfidence * 0.9
        case .complete: return avgConfidence * 0.7
        }
    }
}