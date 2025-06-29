import Foundation
import Vision
import CoreGraphics
import QuartzCore
import AVFoundation

// MARK: - Shot Detection States
enum ShotPhase: String, CaseIterable {
    case ready = "Ready"           // Person in neutral stance
    case preparation = "Loading"   // Arms coming up, preparing to shoot
    case shotRelease = "Release"   // Peak arm extension, shot released
    case followThrough = "Follow"  // Arm extended after release
    case complete = "Complete"     // Shot completed, arms returning
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
    let releasePosition: CGPoint?  // Wrist position at release
    let timestamp: TimeInterval
    let shootingForm: ShootingFormAnalysis?
    let motionQuality: MotionQuality?
}

// MARK: - Motion Quality Analysis
struct MotionQuality {
    let armExtensionVelocity: CGFloat?     // Speed of arm extension
    let legExtensionDetected: Bool         // Any leg drive detected
    let smoothnessScore: Float             // 0-1 score for motion smoothness
    let symmetryScore: Float               // 0-1 score for body symmetry
    let audioFeedback: [String]            // Real-time coaching cues
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
    
    // MARK: - Configuration (Pure Motion-Based)
    private struct Config {
        // REAL SHOOTER thresholds - match user's 47-48° actual form
        static let minimumArmExtensionVelocity: CGFloat = 0.06  // Lower velocity threshold
        static let minimumArmExtensionAngle: CGFloat = 45.0     // Match observed minimum (47°)
        static let preparationArmAngle: CGFloat = 40.0          // Very low for real shooters
        static let releaseArmAngle: CGFloat = 50.0              // Match observed max (48°)
        static let quickReleaseArmAngle: CGFloat = 45.0         // Quick releases (match minimum)
        static let armDownThreshold: CGFloat = 70.0             // Lower threshold for arms down
        
        // Relative position requirements (BALANCED - catch real shots, avoid false positives)
        static let minimumWristAboveShoulderDistance: CGFloat = 0.08  // Wrist above shoulder (more realistic)
        static let minimumElbowAboveShoulderDistance: CGFloat = 0.02  // Elbow slightly above shoulder
        static let preparationWristHeight: CGFloat = 0.55      // Wrist in upper 55% of screen for prep (more lenient)
        static let releaseWristHeight: CGFloat = 0.35          // Wrist in upper 35% of screen for release (more lenient)
        
        // Timing thresholds - OPTIMIZED FOR QUICK SHOOTERS
        static let phaseTimeoutSeconds: TimeInterval = 6.0     // Reset if stuck in phase
        static let minimumPreparationTime: TimeInterval = 0.05 // Very short prep allowed
        static let quickReleaseMaxTime: TimeInterval = 0.8     // Allow direct ready→release within this time
        static let minimumReleaseTime: TimeInterval = 0.05     // Very quick release allowed
        static let followThroughDuration: TimeInterval = 0.8   // Shorter follow-through
        
        // Audio debouncing - prevent overlapping speech with better pacing
        static let minimumAudioInterval: TimeInterval = 3.0    // Slower audio for better clarity
        static let quickShotAudioInterval: TimeInterval = 2.0  // Reasonable pace for quick shots
        static let shotAudioCooldown: TimeInterval = 4.0       // Prevent rapid "shot" announcements
        
        // Hand determination
        static let handMovementThreshold: CGFloat = 0.08       // Hand velocity for dominant detection
        
        // Progressive coaching thresholds - ADJUSTED for real shooting form
        static let goodFormArmAngle: CGFloat = 50.0            // Realistic target (vs 160°)
        static let excellentFormArmAngle: CGFloat = 60.0       // Excellent extension (vs 170°)
        static let goodLegExtensionThreshold: CGFloat = 0.02   // Minimum leg drive
        static let excellentSmoothness: Float = 0.7            // Smooth motion score (more lenient)
    }
    
    // MARK: - Private Properties
    private var currentPhase: ShotPhase = .ready
    private var phaseStartTime: TimeInterval = 0
    private var shotStartTime: TimeInterval = 0
    private var lastPosePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
    private var shootingHand: ShootingHand = .unknown
    private var releasePosition: CGPoint?  // Wrist position at release
    private var shotHistory: [ShotDetectionResult] = []
    
    // Motion tracking for biomechanics analysis
    private var leftArmHistory: [ArmPose] = []
    private var rightArmHistory: [ArmPose] = []
    private var legMotionHistory: [LegPose] = []
    private let motionHistorySize = 10
    
    // Audio feedback system with debouncing
    private var lastAudioFeedback: TimeInterval = 0
    private var lastShotAudioTime: TimeInterval = 0
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    // MARK: - Public Interface
    
    /// Process current frame for pure motion-based shot detection
    func processShotDetection(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool = false,
        currentTime: TimeInterval = CACurrentMediaTime()
    ) -> ShotDetectionResult? {
        
        // CRITICAL: Only process shot detection with fresh pose data
        guard !posePoints.isEmpty else {
            print("[ShotDetection] ❌ NO POSE DATA - cannot detect shots without current pose")
            return nil
        }
        
        // Store current pose data
        lastPosePoints = posePoints
        
        // Update motion history for biomechanics analysis
        updateMotionHistory(posePoints: posePoints, useFrontCamera: useFrontCamera, currentTime: currentTime)
        
        // Determine shooting hand based on motion patterns (not ball)
        if shootingHand == .unknown {
            shootingHand = determineShootingHandFromMotion(posePoints: posePoints, useFrontCamera: useFrontCamera)
        }
        
        // Process pure motion-based state machine
        let newPhase = determineNextPhase(
            currentPhase: currentPhase,
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
        
        // IMPROVED: Phase timeout and loop prevention
        if currentTime - phaseStartTime > Config.phaseTimeoutSeconds {
            if currentPhase != .ready {
                print("[ShotDetection] ⏰ Phase timeout (\(String(format: "%.1f", currentTime - phaseStartTime))s), resetting to ready")
                resetShotDetection()
            }
        }
        
        // CRITICAL: Prevent infinite loops by forcing phase progression
        if currentPhase == .shotRelease && currentTime - phaseStartTime > 2.0 {
            print("[ShotDetection] 🔄 FORCED PROGRESSION: Release→Follow-through (stuck prevention)")
            currentPhase = .followThrough
            phaseStartTime = currentTime
        } else if currentPhase == .followThrough && currentTime - phaseStartTime > 3.0 {
            print("[ShotDetection] 🔄 FORCED PROGRESSION: Follow-through→Complete (stuck prevention)")
            currentPhase = .complete
            phaseStartTime = currentTime
        }
        
        // Analyze shooting form
        let formAnalysis = analyzeShootingForm(
            posePoints: posePoints,
            phase: currentPhase,
            useFrontCamera: useFrontCamera
        )
        
        // Analyze motion quality and generate audio feedback
        let motionQuality = analyzeMotionQuality(posePoints: posePoints, phase: currentPhase, currentTime: currentTime)
        
        // Provide real-time audio feedback
        if let feedback = motionQuality?.audioFeedback, !feedback.isEmpty {
            provideFeedback(feedback, currentTime: currentTime)
        }
        
        // Create result
        let result = ShotDetectionResult(
            phase: currentPhase,
            shootingHand: shootingHand,
            confidence: calculateConfidence(phase: currentPhase, posePoints: posePoints),
            releasePosition: releasePosition,
            timestamp: currentTime,
            shootingForm: formAnalysis,
            motionQuality: motionQuality
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
        leftArmHistory.removeAll()
        rightArmHistory.removeAll()
        legMotionHistory.removeAll()
        lastAudioFeedback = 0
        lastShotAudioTime = 0
    }
    
    /// Reset state when pose is lost (called when no pose detected)
    func handlePoseLost() {
        // Only reset if we're not in a critical shot phase
        if currentPhase != .shotRelease && currentPhase != .followThrough {
            print("[ShotDetection] 🔄 Pose lost - resetting to ready state")
            currentPhase = .ready
            phaseStartTime = CACurrentMediaTime()
            // Don't reset shooting hand determination or audio timing
        }
    }
    
    // MARK: - Private Methods
    
    // Removed - replaced with motion-based hand determination
    
    private func determineNextPhase(
        currentPhase: ShotPhase,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> ShotPhase {
        
        switch currentPhase {
        case .ready:
            // Check for QUICK RELEASE first (skip preparation for fast shooters)
            if let quickRelease = detectQuickRelease(posePoints: posePoints, useFrontCamera: useFrontCamera, currentTime: currentTime) {
                return quickRelease
            }
            // Otherwise check for preparation
            return detectPreparationPhase(posePoints: posePoints, useFrontCamera: useFrontCamera)
            
        case .preparation:
            return detectReleasePhase(posePoints: posePoints, useFrontCamera: useFrontCamera, currentTime: currentTime)
            
        case .shotRelease:
            return detectFollowThroughPhase(posePoints: posePoints, currentTime: currentTime)
            
        case .followThrough:
            return detectCompletionPhase(currentTime: currentTime)
            
        case .complete:
            // Auto-reset after completion
            return .ready
        }
    }
    
    private func detectPreparationPhase(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool
    ) -> ShotPhase {
        
        // Require BOTH proper arm angle AND proper relative positioning
        guard let dominantArmAngle = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand) else {
            return .ready
        }
        
        // Check arm extension with debug info
        if dominantArmAngle <= Config.preparationArmAngle {
            print("[ShotDetection] 📐 Arm angle too low for prep: \(String(format: "%.1f", dominantArmAngle))° (need > \(Config.preparationArmAngle)°)")
            return .ready
        }
        
        // CRITICAL: Check relative arm position to body landmarks
        guard isArmProperlyElevated(posePoints: posePoints, shootingHand: shootingHand, useFrontCamera: useFrontCamera, requiredHeight: Config.preparationWristHeight) else {
            return .ready
        }
        
        print("[ShotDetection] 🏀 PREPARATION: Arms properly elevated (\(String(format: "%.1f", dominantArmAngle))°)")
        return .preparation
    }
    
    /// NEW: Detect quick releases that skip preparation phase
    private func detectQuickRelease(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> ShotPhase? {
        
        // Check if we've been in ready state recently (quick shooter)
        guard currentTime - phaseStartTime < Config.quickReleaseMaxTime else {
            return nil // Been in ready too long, require normal preparation
        }
        
        // Check for strong arm extension OR high velocity (quick release indicators)
        guard let armExtensionAngle = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand) else {
            return nil
        }
        
        let armVelocity = calculateArmExtensionVelocity(posePoints: posePoints, shootingHand: shootingHand)
        
        // QUICK RELEASE CONDITIONS: Lower angle threshold AND high velocity
        let hasQuickExtension = armExtensionAngle > Config.quickReleaseArmAngle
        let hasQuickVelocity = armVelocity > Config.minimumArmExtensionVelocity
        
        if hasQuickExtension && hasQuickVelocity {
            // Also check relative positioning (but more lenient)
            if isArmProperlyElevated(posePoints: posePoints, shootingHand: shootingHand, useFrontCamera: useFrontCamera, requiredHeight: 0.45) {
                // Store release position
                let wristJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
                if let wristPoint = posePoints[wristJoint], wristPoint.confidence > 0.5 {
                    let wristX = useFrontCamera ? (1.0 - wristPoint.x) : wristPoint.x
                    releasePosition = CGPoint(x: wristX, y: 1.0 - wristPoint.y)
                }
                
                print("[ShotDetection] ⚡ QUICK RELEASE: \(String(format: "%.1f", armExtensionAngle))° + \(String(format: "%.3f", armVelocity)) velocity")
                return .shotRelease
            }
        }
        
        return nil // No quick release detected
    }
    
    private func detectReleasePhase(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> ShotPhase {
        
        // Require BOTH strong arm extension AND proper elevation for release
        guard let armExtensionAngle = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand) else {
            return .preparation
        }
        
        // Check arm extension for release with debug
        if armExtensionAngle <= Config.releaseArmAngle {
            print("[ShotDetection] 📐 Arm angle too low for release: \(String(format: "%.1f", armExtensionAngle))° (need > \(Config.releaseArmAngle)°)")
            return .preparation
        }
        
        // CRITICAL: Arms must be highly elevated for release (upper 25% of screen)
        guard isArmProperlyElevated(posePoints: posePoints, shootingHand: shootingHand, useFrontCamera: useFrontCamera, requiredHeight: Config.releaseWristHeight) else {
            return .preparation
        }
        
        // Allow releases without minimum prep time (handled by quick release detection)
        
        // Check for arm extension velocity (additional validation)
        let armVelocity = calculateArmExtensionVelocity(posePoints: posePoints, shootingHand: shootingHand)
        
        // Store release position (wrist location)
        let wristJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        if let wristPoint = posePoints[wristJoint], wristPoint.confidence > 0.5 {
            let wristX = useFrontCamera ? (1.0 - wristPoint.x) : wristPoint.x
            releasePosition = CGPoint(x: wristX, y: 1.0 - wristPoint.y)
        }
        
        print("[ShotDetection] 🚀 RELEASE: Strong extension (\(String(format: "%.1f", armExtensionAngle))°) + proper elevation")
        return .shotRelease
    }
    
    private func detectFollowThroughPhase(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        currentTime: TimeInterval
    ) -> ShotPhase {
        
        // FIXED: Ensure proper transition from release to follow-through
        // Check minimum time in release phase first
        if currentTime - phaseStartTime < Config.minimumReleaseTime {
            return .shotRelease  // Stay in release for minimum time
        }
        
        // Check if we should move to completion (timeout or arm lowered)
        if currentTime - phaseStartTime > Config.followThroughDuration {
            print("[ShotDetection] ✅ AUTO-COMPLETE: Follow-through timeout")
            return .complete
        }
        
        // Check for arm lowering (shot completion)
        if let armExtension = calculateArmExtension(posePoints: posePoints, shootingHand: shootingHand) {
            if armExtension < Config.armDownThreshold {
                print("[ShotDetection] ✅ FOLLOW-THROUGH→COMPLETE: Arm lowered to \(String(format: "%.1f", armExtension))°")
                return .complete
            }
        }
        
        // Check for proper follow-through form (arm extended, wrist snapped)
        let shootingElbow: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftElbow : .rightElbow
        let shootingWrist: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        guard let elbowPoint = posePoints[shootingElbow],
              let wristPoint = posePoints[shootingWrist],
              elbowPoint.confidence > 0.5, wristPoint.confidence > 0.5 else {
            // If we can't detect arm position, auto-advance after reasonable time
            if currentTime - phaseStartTime > 1.0 {
                return .complete
            }
            return .followThrough
        }
        
        // Continue follow-through if arm still extended
        if wristPoint.y < elbowPoint.y {  // Wrist higher than elbow = extended arm
            print("[ShotDetection] 📐 FOLLOW-THROUGH: Arm still extended")
            return .followThrough
        }
        
        // Arm has lowered - shot complete
        print("[ShotDetection] ✅ FOLLOW-THROUGH→COMPLETE: Arm position changed")
        return .complete
    }
    
    private func detectCompletionPhase(currentTime: TimeInterval) -> ShotPhase {
        // ALWAYS complete after reasonable timeout to prevent loops
        if currentTime - phaseStartTime > Config.followThroughDuration {
            print("[ShotDetection] ✅ SHOT COMPLETE: Auto-timeout (\(String(format: "%.1f", currentTime - phaseStartTime))s)")
            return .complete
        }
        
        // Check if arm has returned to non-extended position (shot truly complete)
        if let armAngle = calculateArmExtension(posePoints: lastPosePoints, shootingHand: shootingHand) {
            // Shot complete when arm comes back down (less extended)
            if armAngle < Config.armDownThreshold && currentTime - phaseStartTime > 0.5 {
                print("[ShotDetection] ✅ SHOT COMPLETE: Arm returned down (\(String(format: "%.1f", armAngle))°)")
                return .complete
            }
        }
        
        // Stay in follow-through briefly, then auto-complete
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
    
    // MARK: - Motion Analysis Methods
    
    private func updateMotionHistory(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) {
        // Update left arm history
        if let leftArmPose = createArmPose(posePoints: posePoints, side: .left, useFrontCamera: useFrontCamera, currentTime: currentTime) {
            leftArmHistory.append(leftArmPose)
            if leftArmHistory.count > motionHistorySize {
                leftArmHistory.removeFirst()
            }
        }
        
        // Update right arm history
        if let rightArmPose = createArmPose(posePoints: posePoints, side: .right, useFrontCamera: useFrontCamera, currentTime: currentTime) {
            rightArmHistory.append(rightArmPose)
            if rightArmHistory.count > motionHistorySize {
                rightArmHistory.removeFirst()
            }
        }
        
        // Update leg motion history
        if let legPose = createLegPose(posePoints: posePoints, useFrontCamera: useFrontCamera, currentTime: currentTime) {
            legMotionHistory.append(legPose)
            if legMotionHistory.count > motionHistorySize {
                legMotionHistory.removeFirst()
            }
        }
    }
    
    private func createArmPose(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        side: ShootingHand,
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> ArmPose? {
        let shoulder: VNHumanBodyPoseObservation.JointName = side == .left ? .leftShoulder : .rightShoulder
        let elbow: VNHumanBodyPoseObservation.JointName = side == .left ? .leftElbow : .rightElbow
        let wrist: VNHumanBodyPoseObservation.JointName = side == .left ? .leftWrist : .rightWrist
        
        guard let shoulderPoint = posePoints[shoulder],
              let elbowPoint = posePoints[elbow],
              let wristPoint = posePoints[wrist],
              shoulderPoint.confidence > 0.5,
              elbowPoint.confidence > 0.5,
              wristPoint.confidence > 0.5 else { return nil }
        
        // Convert coordinates
        let shoulderPos = convertPosePoint(shoulderPoint, useFrontCamera: useFrontCamera)
        let elbowPos = convertPosePoint(elbowPoint, useFrontCamera: useFrontCamera)
        let wristPos = convertPosePoint(wristPoint, useFrontCamera: useFrontCamera)
        
        // Calculate extension angle
        let extensionAngle = calculateAngleBetweenPoints(p1: shoulderPos, p2: elbowPos, p3: wristPos)
        
        // Calculate velocity (if we have previous pose)
        var velocity: CGVector = .zero
        let armHistory = side == .left ? leftArmHistory : rightArmHistory
        if let lastPose = armHistory.last {
            let deltaTime = currentTime - lastPose.timestamp
            if deltaTime > 0 {
                velocity = CGVector(
                    dx: (wristPos.x - lastPose.wristPos.x) / CGFloat(deltaTime),
                    dy: (wristPos.y - lastPose.wristPos.y) / CGFloat(deltaTime)
                )
            }
        }
        
        return ArmPose(
            shoulderPos: shoulderPos,
            elbowPos: elbowPos,
            wristPos: wristPos,
            extensionAngle: extensionAngle,
            velocity: velocity,
            timestamp: currentTime
        )
    }
    
    private func createLegPose(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool,
        currentTime: TimeInterval
    ) -> LegPose? {
        // Use right leg as representative (could extend to both legs)
        guard let hipPoint = posePoints[.rightHip],
              let kneePoint = posePoints[.rightKnee],
              hipPoint.confidence > 0.5,
              kneePoint.confidence > 0.5 else { return nil }
        
        let hipPos = convertPosePoint(hipPoint, useFrontCamera: useFrontCamera)
        let kneePos = convertPosePoint(kneePoint, useFrontCamera: useFrontCamera)
        
        // Try to get ankle for full leg analysis
        var anklePos: CGPoint?
        var extensionAngle: CGFloat?
        
        if let anklePoint = posePoints[.rightAnkle], anklePoint.confidence > 0.4 {
            anklePos = convertPosePoint(anklePoint, useFrontCamera: useFrontCamera)
            extensionAngle = calculateAngleBetweenPoints(p1: hipPos, p2: kneePos, p3: anklePos!)
        }
        
        return LegPose(
            hipPos: hipPos,
            kneePos: kneePos,
            anklePos: anklePos,
            extensionAngle: extensionAngle,
            timestamp: currentTime
        )
    }
    
    private func convertPosePoint(_ point: VNRecognizedPoint, useFrontCamera: Bool) -> CGPoint {
        let x = useFrontCamera ? (1.0 - point.x) : point.x
        let y = 1.0 - point.y
        return CGPoint(x: x, y: y)
    }
    
    private func calculateAngleBetweenPoints(p1: CGPoint, p2: CGPoint, p3: CGPoint) -> CGFloat {
        let v1 = CGVector(dx: p1.x - p2.x, dy: p1.y - p2.y)
        let v2 = CGVector(dx: p3.x - p2.x, dy: p3.y - p2.y)
        
        let dot = v1.dx * v2.dx + v1.dy * v2.dy
        let mag1 = sqrt(v1.dx * v1.dx + v1.dy * v1.dy)
        let mag2 = sqrt(v2.dx * v2.dx + v2.dy * v2.dy)
        
        guard mag1 > 0, mag2 > 0 else { return 0 }
        
        let cosAngle = dot / (mag1 * mag2)
        let angle = acos(max(-1, min(1, cosAngle))) * 180 / .pi
        
        return angle
    }
    
    /// CRITICAL: Check if arm is properly elevated relative to body landmarks
    private func isArmProperlyElevated(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        shootingHand: ShootingHand,
        useFrontCamera: Bool,
        requiredHeight: CGFloat
    ) -> Bool {
        
        let shoulderJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftShoulder : .rightShoulder
        let elbowJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftElbow : .rightElbow
        let wristJoint: VNHumanBodyPoseObservation.JointName = shootingHand == .left ? .leftWrist : .rightWrist
        
        guard let shoulderPoint = posePoints[shoulderJoint],
              let elbowPoint = posePoints[elbowJoint],
              let wristPoint = posePoints[wristJoint],
              shoulderPoint.confidence > 0.5,
              elbowPoint.confidence > 0.5,
              wristPoint.confidence > 0.5 else {
            return false
        }
        
        // Convert to screen coordinates
        let shoulderY = 1.0 - shoulderPoint.y
        let elbowY = 1.0 - elbowPoint.y
        let wristY = 1.0 - wristPoint.y
        
        // CHECK 1: Wrist must be in upper portion of screen
        guard wristY < requiredHeight else {
            print("[ShotDetection] ❌ Arm not elevated enough: wrist at \(String(format: "%.3f", wristY)), need < \(requiredHeight)")
            return false
        }
        
        // CHECK 2: Wrist must be significantly above shoulder
        let wristAboveShoulder = shoulderY - wristY
        guard wristAboveShoulder > Config.minimumWristAboveShoulderDistance else {
            print("[ShotDetection] ❌ Wrist not above shoulder: distance \(String(format: "%.3f", wristAboveShoulder))")
            return false
        }
        
        // CHECK 3: Elbow should be at or above shoulder level
        let elbowAboveShoulder = shoulderY - elbowY
        guard elbowAboveShoulder > -Config.minimumElbowAboveShoulderDistance else {
            print("[ShotDetection] ❌ Elbow too low: distance \(String(format: "%.3f", elbowAboveShoulder))")
            return false
        }
        
        print("[ShotDetection] ✅ ARM PROPERLY ELEVATED: wrist=\(String(format: "%.3f", wristY)), above shoulder by \(String(format: "%.3f", wristAboveShoulder))")
        return true
    }
    
    private func calculateArmExtensionVelocity(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        shootingHand: ShootingHand
    ) -> CGFloat {
        let armHistory = shootingHand == .left ? leftArmHistory : rightArmHistory
        
        guard armHistory.count >= 2 else { return 0 }
        
        let recent = Array(armHistory.suffix(2))
        let deltaTime = recent[1].timestamp - recent[0].timestamp
        guard deltaTime > 0 else { return 0 }
        
        let deltaAngle = recent[1].extensionAngle - recent[0].extensionAngle
        return abs(deltaAngle / CGFloat(deltaTime))
    }
    
    private func determineShootingHandFromMotion(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        useFrontCamera: Bool
    ) -> ShootingHand {
        // Determine shooting hand based on which arm shows more upward motion
        let leftVelocity = leftArmHistory.last?.velocity.dy ?? 0
        let rightVelocity = rightArmHistory.last?.velocity.dy ?? 0
        
        // Shooting hand typically moves upward faster (negative Y = up)
        if leftVelocity < rightVelocity - Config.handMovementThreshold {
            return .left
        } else if rightVelocity < leftVelocity - Config.handMovementThreshold {
            return .right
        }
        
        // Fallback: use arm extension angle
        let leftExtension = leftArmHistory.last?.extensionAngle ?? 0
        let rightExtension = rightArmHistory.last?.extensionAngle ?? 0
        
        return leftExtension > rightExtension ? .left : .right
    }
    
    private func analyzeMotionQuality(
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint],
        phase: ShotPhase,
        currentTime: TimeInterval
    ) -> MotionQuality? {
        
        guard phase != .ready else { return nil }
        
        let armHistory = shootingHand == .left ? leftArmHistory : rightArmHistory
        
        // Calculate motion metrics
        let armVelocity = armHistory.last?.velocity.dy ?? 0
        let legExtension = detectLegExtension()
        let smoothness = calculateMotionSmoothness(armHistory: armHistory)
        let symmetry = calculateBodySymmetry(posePoints: posePoints)
        
        // Generate coaching feedback based on current phase and motion quality
        var audioFeedback: [String] = []
        
        // Progressive coaching based on skill level detection
        let armAngle = armHistory.last?.extensionAngle ?? 0
        let isGoodForm = armAngle > Config.goodFormArmAngle
        let isExcellentForm = armAngle > Config.excellentFormArmAngle
        
        switch phase {
        case .preparation:
            // Rich preparation feedback with variety
            let prepFeedback = [
                "Loading up", "Getting ready", "Here we go", "Setting up", "Good preparation"
            ]
            audioFeedback.append(prepFeedback.randomElement() ?? "Loading")
            
            // Specific coaching based on detection
            if !legExtension {
                let legTips = [
                    "Power through your legs", "Drive from the ground up", "Use those legs", 
                    "Leg power", "Push through your feet"
                ]
                audioFeedback.append(legTips.randomElement() ?? "Power from legs")
            }
            
        case .shotRelease:
            // VARIED SHOT DETECTION - Rich, creative responses
            let shotResponses = [
                "Shot detected!", "There it goes!", "Release!", "Fired!", 
                "Shot away!", "Let it fly!", "Up and away!", "Shot released!",
                "Money!", "Shooter!", "Bang!", "From downtown!", "Splash!"
            ]
            audioFeedback.append(shotResponses.randomElement() ?? "Shot!")
            
            // Rich form analysis with creative feedback
            if isExcellentForm && legExtension && smoothness > Config.excellentSmoothness {
                let excellentFeedback = [
                    "Perfect mechanics, Kobe would be proud!", "Textbook form!", 
                    "NBA-level shot!", "Pure shooter!", "Flawless technique!",
                    "That's championship form!", "Smooth as silk!"
                ]
                audioFeedback.append(excellentFeedback.randomElement() ?? "Perfect!")
            } else if isGoodForm && legExtension {
                let goodFeedback = [
                    "Solid fundamentals!", "Great mechanics!", "Nice form!",
                    "Professional technique!", "Clean shot!", "Smooth release!"
                ]
                audioFeedback.append(goodFeedback.randomElement() ?? "Good form!")
            } else {
                // Specific improvement areas with encouragement
                if !legExtension && armAngle < Config.goodFormArmAngle {
                    let improvementTips = [
                        "Drive through legs and extend higher", "Power up and reach for the sky",
                        "Legs to fingertips - full extension", "Ground force to high arc"
                    ]
                    audioFeedback.append(improvementTips.randomElement() ?? "Use legs and extend")
                } else if !legExtension {
                    let legReminders = [
                        "Remember your leg drive", "Power comes from the ground",
                        "Push through those legs", "Leg power, shooter!"
                    ]
                    audioFeedback.append(legReminders.randomElement() ?? "Use those legs")
                } else if armAngle < Config.goodFormArmAngle {
                    let extensionTips = [
                        "Reach for the clouds", "Higher arc needed", "Extend that follow-through",
                        "Get that rainbow arc", "Shoot for the sky"
                    ]
                    audioFeedback.append(extensionTips.randomElement() ?? "Extend higher")
                }
            }
            
        case .followThrough:
            let followThroughFeedback = [
                "Nice follow-through", "Hold that form", "Beautiful extension",
                "Perfect finish", "Hold it high", "Good wrist snap"
            ]
            audioFeedback.append(followThroughFeedback.randomElement() ?? "Good follow-through")
            
        case .complete:
            // Rich summary feedback with personality
            if isExcellentForm && legExtension && smoothness > Config.excellentSmoothness {
                let masterFeedback = [
                    "That's how it's done!", "Mamba mentality right there!", 
                    "Perfection in motion!", "Shot of the day!", "Elite level shooting!",
                    "You're in the zone!", "Pure basketball poetry!"
                ]
                audioFeedback.append(masterFeedback.randomElement() ?? "Excellent!")
            } else if isGoodForm && legExtension {
                let solidFeedback = [
                    "Keep that consistency!", "Solid fundamentals!", "Building good habits!",
                    "Nice rhythm!", "That's the way!", "Good work!"
                ]
                audioFeedback.append(solidFeedback.randomElement() ?? "Good shot!")
            } else {
                // Encouraging improvement guidance
                let nextShotTips = [
                    "Next rep, focus on your legs", "Remember: legs power the shot",
                    "Think legs to fingertips", "Drive from the ground up next time",
                    "Power through your foundation"
                ]
                audioFeedback.append(nextShotTips.randomElement() ?? "Keep improving!")
            }
            
        default:
            break
        }
        
        return MotionQuality(
            armExtensionVelocity: abs(armVelocity),
            legExtensionDetected: legExtension,
            smoothnessScore: smoothness,
            symmetryScore: symmetry,
            audioFeedback: audioFeedback
        )
    }
    
    private func detectLegExtension() -> Bool {
        guard legMotionHistory.count >= 2 else { return false }
        
        let recent = Array(legMotionHistory.suffix(2))
        
        // Detect if knees are extending (knee moving away from hip)
        let hipKneeDistance1 = distance(from: recent[0].hipPos, to: recent[0].kneePos)
        let hipKneeDistance2 = distance(from: recent[1].hipPos, to: recent[1].kneePos)
        
        return hipKneeDistance2 > hipKneeDistance1 + 0.02  // Legs extending
    }
    
    private func calculateMotionSmoothness(armHistory: [ArmPose]) -> Float {
        guard armHistory.count >= 3 else { return 1.0 }
        
        var accelerationChanges: [CGFloat] = []
        
        for i in 2..<armHistory.count {
            let v1 = armHistory[i-1].velocity.dy
            let v2 = armHistory[i].velocity.dy
            let acceleration = v2 - v1
            
            if i > 2 {
                let prevAcceleration = armHistory[i-1].velocity.dy - armHistory[i-2].velocity.dy
                accelerationChanges.append(abs(acceleration - prevAcceleration))
            }
        }
        
        guard !accelerationChanges.isEmpty else { return 1.0 }
        
        let avgChange = accelerationChanges.reduce(0, +) / CGFloat(accelerationChanges.count)
        return max(0, min(1, 1.0 - Float(avgChange * 10)))  // Lower change = smoother
    }
    
    private func calculateBodySymmetry(posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]) -> Float {
        // Simple symmetry check: compare shoulder heights
        guard let leftShoulder = posePoints[.leftShoulder],
              let rightShoulder = posePoints[.rightShoulder],
              leftShoulder.confidence > 0.5,
              rightShoulder.confidence > 0.5 else { return 0.5 }
        
        let shoulderDifference = abs(leftShoulder.y - rightShoulder.y)
        return max(0, min(1, 1.0 - Float(shoulderDifference * 5)))  // Lower difference = better symmetry
    }
    
    private func provideFeedback(_ feedback: [String], currentTime: TimeInterval) {
        guard let firstFeedback = feedback.first else { return }
        
        // SMART DEBOUNCING: Different intervals for different feedback types
        let isShot = firstFeedback.contains("Shot") || firstFeedback.contains("!")
        let timeSinceLastAudio = currentTime - lastAudioFeedback
        let timeSinceLastShot = currentTime - lastShotAudioTime
        
        if isShot {
            // Quick shot detection: allow faster audio for shots
            guard timeSinceLastShot > Config.quickShotAudioInterval else {
                print("[ShotDetection] 🔇 Shot audio debounced (too soon: \(String(format: "%.1f", timeSinceLastShot))s)")
                return
            }
            lastShotAudioTime = currentTime
        } else {
            // Regular feedback: longer debouncing
            guard timeSinceLastAudio > Config.minimumAudioInterval else {
                print("[ShotDetection] 🔇 Audio debounced (too soon: \(String(format: "%.1f", timeSinceLastAudio))s)")
                return
            }
        }
        
        print("[ShotDetection] 🎤 AUDIO: \(firstFeedback)")
        
        // Stop any currently playing speech to prevent overlap
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        
        // Provide real-time audio feedback using speech synthesis with better voice
        let utterance = AVSpeechUtterance(string: firstFeedback)
        utterance.rate = 0.5  // SLOWER for clarity and impact
        utterance.volume = 1.0  // Full volume
        utterance.pitchMultiplier = 0.9  // Slightly lower pitch for authority
        
        // Try to find a male voice for more coaching feel
        let preferredVoices = [
            "com.apple.ttsbundle.Daniel-compact",  // Male voice
            "com.apple.voice.compact.en-US.Zarvox", // Robot-like but clear
            "com.apple.ttsbundle.siri_male_en-US_compact" // Siri male
        ]
        
        var selectedVoice: AVSpeechSynthesisVoice?
        for voiceId in preferredVoices {
            if let voice = AVSpeechSynthesisVoice(identifier: voiceId) {
                selectedVoice = voice
                break
            }
        }
        
        // Fallback to any available male voice or default
        if selectedVoice == nil {
            selectedVoice = AVSpeechSynthesisVoice.speechVoices().first { voice in
                voice.language.hasPrefix("en") && voice.gender == .male
            } ?? AVSpeechSynthesisVoice(language: "en-US")
        }
        
        utterance.voice = selectedVoice
        
        speechSynthesizer.speak(utterance)
        lastAudioFeedback = currentTime
    }
    
    // Removed - replaced with more comprehensive motion tracking
    
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
        phase: ShotPhase,
        useFrontCamera: Bool
    ) -> ShootingFormAnalysis? {
        
        // Only analyze form during key phases
        guard phase == .preparation || phase == .shotRelease || phase == .followThrough else { return nil }
        
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
        case .shotRelease: return avgConfidence * 1.0
        case .followThrough: return avgConfidence * 0.9
        case .complete: return avgConfidence * 0.7
        }
    }
}

// MARK: - Motion Tracking Structures
struct ArmPose {
    let shoulderPos: CGPoint
    let elbowPos: CGPoint
    let wristPos: CGPoint
    let extensionAngle: CGFloat
    let velocity: CGVector
    let timestamp: TimeInterval
}

struct LegPose {
    let hipPos: CGPoint
    let kneePos: CGPoint
    let anklePos: CGPoint?
    let extensionAngle: CGFloat?
    let timestamp: TimeInterval
}