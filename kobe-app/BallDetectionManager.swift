import Foundation
import CoreImage
import UIKit
import Vision

// MARK: - Ball Detection Result
struct BallDetectionResult {
    let position: CGPoint
    let boundingBox: CGRect
    let confidence: Float
    let velocity: CGVector
    let timestamp: TimeInterval
    
    static func create(from detection: Detection, velocity: CGVector = .zero) -> BallDetectionResult {
        return BallDetectionResult(
            position: CGPoint(x: detection.boundingBox.midX, y: detection.boundingBox.midY),
            boundingBox: detection.boundingBox,
            confidence: detection.confidence,
            velocity: velocity,
            timestamp: CACurrentMediaTime()
        )
    }
}

// MARK: - Motion-based Ball Detection Manager
class BallDetectionManager {
    
    // MARK: - Detection Configuration
    private struct Config {
        static let minimumConfidenceFrames = 1  // Immediate detection for fast balls
        static let maxTrackingHistory = 5      // Reduced for faster response
        static let minimumCircularity: Float = 0.6  // More strict for mini balls
        static let maximumCircularity: Float = 1.8  // Less lenient to avoid false positives
        static let minimumArea: CGFloat = 0.0005  // Slightly larger minimum
        static let maximumArea: CGFloat = 0.15   // Much smaller max to avoid body detections
        static let velocityThreshold: CGFloat = 0.005  // Lower threshold for mini balls
        static let maxMotionJump: CGFloat = 0.25     // Allow bigger jumps for fast balls
        static let temporalSmoothingFactor: Float = 0.1  // Much less smoothing for responsiveness
        static let handProximityThreshold: CGFloat = 0.15  // Ball must be within this distance of hands
        static let minimumBallConfidence: Float = 0.15     // Much higher confidence required
        static let handProximityBonus: Float = 0.3        // Bonus confidence for balls near hands
    }
    
    // MARK: - Private Properties
    private var detectionHistory: [BallDetectionResult] = []
    private var confirmedDetections: [BallDetectionResult] = []
    private var lastValidPosition: CGPoint?
    private var currentVelocity: CGVector = .zero
    private var motionTracker: OpticalFlowTracker?
    private var frameCounter: Int = 0
    
    // MARK: - Public Interface
    
    /// Process current frame and return confident ball detections
    func processBallDetections(
        _ detections: [Detection],
        personBoundingBox: CGRect?,
        posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]? = nil,
        useFrontCamera: Bool = false,
        shotInProgress: Bool = false,
        currentTime: TimeInterval = CACurrentMediaTime()
    ) -> BallDetectionResult? {
        frameCounter += 1
        
        // Debug: Log all incoming detections (reduced frequency)
        let basketballs = detections.filter { $0.label == "basketball" }
        if basketballs.count > 0 && frameCounter % 30 == 0 {
            print("[BallDetectionManager] 🏀 Processing \(basketballs.count) basketball detections")
        }
        
        // Filter potential ball detections using multiple criteria
        let ballCandidates = detections
            .filter { $0.label == "basketball" }
            .compactMap { validateBallCandidate($0, personBoundingBox: personBoundingBox, posePoints: posePoints, useFrontCamera: useFrontCamera, shotInProgress: shotInProgress) }
        
        guard !ballCandidates.isEmpty else {
            // No ball candidates found, but try motion-based detection
            return processMotionBasedDetection(currentTime: currentTime)
        }
        
        // Select best candidate using multi-factor scoring
        let bestCandidate = selectBestCandidate(from: ballCandidates, currentTime: currentTime)
        
        // Skip heavy temporal filtering for immediate response - just validate motion consistency
        return validateAndReturnBestCandidate(bestCandidate, currentTime: currentTime)
    }
    
    /// Get current ball velocity for physics validation
    var ballVelocity: CGVector {
        return currentVelocity
    }
    
    /// Get trajectory points for visualization
    var trajectoryPoints: [CGPoint] {
        return confirmedDetections.map { $0.position }
    }
    
    /// Clear all tracking data (useful when switching cameras or resetting)
    func clearTrackingData() {
        detectionHistory.removeAll()
        confirmedDetections.removeAll()
        lastValidPosition = nil
        currentVelocity = .zero
        motionTracker = nil
        frameCounter = 0
    }
    
    // MARK: - Private Detection Methods
    
    private func validateBallCandidate(_ detection: Detection, personBoundingBox: CGRect?, posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]?, useFrontCamera: Bool, shotInProgress: Bool) -> BallDetectionResult? {
        let boundingBox = detection.boundingBox
        
        // 0. Minimum confidence check - reject very low confidence detections immediately
        guard detection.confidence >= Config.minimumBallConfidence else {
            return nil
        }
        
        // 1. Hand proximity check - CRITICAL for mini ball detection
        // EXCEPTION: During shot (ball in flight), skip hand proximity check
        let (isNearHands, handProximity) = checkHandProximity(boundingBox, posePoints: posePoints, useFrontCamera: useFrontCamera)
        
        if !shotInProgress {
            // Normal mode: ball must be near hands
            guard isNearHands else {
                return nil
            }
        } else {
            // Shot mode: track ball even if not near hands (ball in flight)
            if frameCounter % 30 == 0 && !isNearHands {
                print("[BallDetectionManager] 🏀 Tracking ball in flight (distance from hands: \(String(format: "%.3f", handProximity)))")
            }
        }
        
        // 2. Size validation using pose-based scaling
        guard isValidBallSize(boundingBox, personBoundingBox: personBoundingBox, posePoints: posePoints) else {
            return nil
        }
        
        // 3. Shape validation (circularity)
        guard isCircularShape(boundingBox) else {
            return nil
        }
        
        // 3. Basic bounds checking - clamp instead of reject for edge cases
        let clampedBox = CGRect(
            x: max(0, min(1, boundingBox.minX)),
            y: max(0, min(1, boundingBox.minY)),
            width: min(boundingBox.width, 1 - max(0, boundingBox.minX)),
            height: min(boundingBox.height, 1 - max(0, boundingBox.minY))
        )
        
        // Only reject if the clamped box is too small to be useful
        guard clampedBox.width > 0.01 && clampedBox.height > 0.01 else {
            return nil
        }
        
        // Calculate velocity if we have previous position
        let velocity = calculateVelocity(for: CGPoint(x: clampedBox.midX, y: clampedBox.midY))
        
        // Boost confidence for balls near hands
        let boostedConfidence = min(1.0, detection.confidence + (isNearHands ? Config.handProximityBonus : 0.0))
        
        // Create detection with clamped bounding box and boosted confidence
        let clampedDetection = Detection(
            boundingBox: clampedBox,
            confidence: boostedConfidence,
            label: detection.label,
            isPerson: detection.isPerson,
            keypoints: detection.keypoints,
            trajectory: detection.trajectory
        )
        
        if frameCounter % 30 == 0 {  // Only log every 30th acceptance
            print("[BallDetectionManager] ✅🏀 MINI BALL ACCEPTED")
        }
        return BallDetectionResult.create(from: clampedDetection, velocity: velocity)
    }
    
    // Check if ball is near hands - CRITICAL validation for mini balls
    private func checkHandProximity(_ boundingBox: CGRect, posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]?, useFrontCamera: Bool) -> (Bool, CGFloat) {
        guard let posePoints = posePoints else {
            print("[BallDetectionManager] No pose points available for hand proximity check")
            return (false, 999.0)  // No pose data = not near hands
        }
        
        let ballCenter = CGPoint(x: boundingBox.midX, y: boundingBox.midY)
        
        // Check both hands
        let handsToCheck: [VNHumanBodyPoseObservation.JointName] = [.leftWrist, .rightWrist]
        var minDistance: CGFloat = 999.0
        var handsDetected = 0
        
        for handJoint in handsToCheck {
            if let handPoint = posePoints[handJoint], handPoint.confidence > 0.5 {
                handsDetected += 1
                // Apply same coordinate transformation as in CameraView
                let handX = useFrontCamera ? (1.0 - handPoint.x) : handPoint.x
                let handY = 1.0 - handPoint.y  // Flip Y coordinate for screen space
                let handPosition = CGPoint(x: handX, y: handY)
                
                let distance = sqrt(pow(ballCenter.x - handPosition.x, 2) + pow(ballCenter.y - handPosition.y, 2))
                minDistance = min(minDistance, distance)
                
                if distance <= Config.handProximityThreshold {
                    if frameCounter % 30 == 0 {
                        print("[BallDetectionManager] ✋ Ball near \(handJoint)")
                    }
                    return (true, distance)
                }
            }
        }
        
        return (false, minDistance)
    }
    
    private func isValidBallSize(_ boundingBox: CGRect, personBoundingBox: CGRect?, posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]?) -> Bool {
        let area = boundingBox.width * boundingBox.height
        
        print("[BallDetectionManager] Size check: area=\(area), min=\(Config.minimumArea), max=\(Config.maximumArea)")
        
        // Basic area constraints
        guard area >= Config.minimumArea && area <= Config.maximumArea else {
            print("[BallDetectionManager] Size rejected: area \(area) outside bounds [\(Config.minimumArea), \(Config.maximumArea)]")
            return false
        }
        
        // Use pose-based scaling if available (more accurate than full person box)
        if let posePoints = posePoints, let handSize = calculateHandSize(from: posePoints) {
            let ballToHandRatio = area / handSize
            
            // Mini ball should be 0.5x to 4x the size of a hand area
            let minHandRatio: CGFloat = 0.3    // Mini ball smaller than hand
            let maxHandRatio: CGFloat = 8.0    // Full basketball much larger than hand
            
            print("[BallDetectionManager] Hand-based ratio check: ratio=\(ballToHandRatio), handSize=\(handSize), min=\(minHandRatio), max=\(maxHandRatio)")
            
            let handRatioValid = ballToHandRatio >= minHandRatio && ballToHandRatio <= maxHandRatio
            if !handRatioValid {
                print("[BallDetectionManager] Size rejected: hand ratio \(ballToHandRatio) outside bounds [\(minHandRatio), \(maxHandRatio)]")
            }
            return handRatioValid
        }
        
        // Fallback to person bounding box if no pose data
        if let personBox = personBoundingBox {
            let personArea = personBox.width * personBox.height
            let ballToPersonRatio = area / personArea
            
            // Much wider range to support mini balls (apple-sized) to full basketballs
            let minRatio: CGFloat = 1.0 / 500.0  // Very small mini balls
            let maxRatio: CGFloat = 1.0 / 8.0    // Full size basketballs
            
            print("[BallDetectionManager] Person-based ratio check: ratio=\(ballToPersonRatio), min=\(minRatio), max=\(maxRatio)")
            
            let ratioValid = ballToPersonRatio >= minRatio && ballToPersonRatio <= maxRatio
            if !ratioValid {
                print("[BallDetectionManager] Size rejected: person ratio \(ballToPersonRatio) outside bounds [\(minRatio), \(maxRatio)]")
            }
            return ratioValid
        }
        
        print("[BallDetectionManager] Size accepted: no reference available")
        return true
    }
    
    // Calculate hand size from pose points for better ball scaling
    private func calculateHandSize(from posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]) -> CGFloat? {
        // Try to get hand-to-wrist or wrist-to-elbow distance for scale reference
        let handJoints: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            (.leftWrist, .leftElbow),    // Left forearm length
            (.rightWrist, .rightElbow),  // Right forearm length
        ]
        
        for (joint1, joint2) in handJoints {
            if let point1 = posePoints[joint1], let point2 = posePoints[joint2],
               point1.confidence > 0.5, point2.confidence > 0.5 {
                
                let distance = sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2))
                
                // Forearm is roughly 3-4x hand length, so hand area ≈ (forearm/3.5)²
                let estimatedHandLength = distance / 3.5
                let estimatedHandArea = estimatedHandLength * estimatedHandLength
                
                print("[BallDetectionManager] Hand size estimated from \(joint1)-\(joint2): \(estimatedHandArea)")
                return estimatedHandArea
            }
        }
        
        // Fallback: Try shoulder width for scale reference
        if let leftShoulder = posePoints[.leftShoulder], let rightShoulder = posePoints[.rightShoulder],
           leftShoulder.confidence > 0.5, rightShoulder.confidence > 0.5 {
            
            let shoulderWidth = abs(leftShoulder.x - rightShoulder.x)
            // Hand is roughly 1/5 of shoulder width
            let estimatedHandLength = shoulderWidth / 5.0
            let estimatedHandArea = estimatedHandLength * estimatedHandLength
            
            print("[BallDetectionManager] Hand size estimated from shoulder width: \(estimatedHandArea)")
            return estimatedHandArea
        }
        
        return nil
    }
    
    private func isCircularShape(_ boundingBox: CGRect) -> Bool {
        let aspectRatio = Float(boundingBox.width / boundingBox.height)
        return aspectRatio >= Config.minimumCircularity && aspectRatio <= Config.maximumCircularity
    }
    
    private func calculateVelocity(for position: CGPoint) -> CGVector {
        guard let lastPosition = lastValidPosition,
              let lastDetection = detectionHistory.last else {
            return .zero
        }
        
        let deltaTime = CACurrentMediaTime() - lastDetection.timestamp
        guard deltaTime > 0 else { return .zero }
        
        let deltaX = position.x - lastPosition.x
        let deltaY = position.y - lastPosition.y
        
        return CGVector(
            dx: deltaX / CGFloat(deltaTime),
            dy: deltaY / CGFloat(deltaTime)
        )
    }
    
    private func selectBestCandidate(from candidates: [BallDetectionResult], currentTime: TimeInterval) -> BallDetectionResult? {
        guard !candidates.isEmpty else { return nil }
        
        // If only one candidate, use it
        if candidates.count == 1 {
            return candidates.first
        }
        
        // Score candidates based on multiple factors
        let scoredCandidates = candidates.map { candidate in
            var score: Float = candidate.confidence
            
            // Bonus for motion consistency
            if let lastVelocity = detectionHistory.last?.velocity {
                let velocityDiff = abs(candidate.velocity.dx - lastVelocity.dx) + abs(candidate.velocity.dy - lastVelocity.dy)
                if velocityDiff < 0.1 {
                    score += 0.2  // Bonus for consistent motion
                }
            }
            
            // Bonus for position consistency
            if let lastPosition = lastValidPosition {
                let distance = sqrt(pow(candidate.position.x - lastPosition.x, 2) + pow(candidate.position.y - lastPosition.y, 2))
                if distance < Config.maxMotionJump {
                    score += 0.1  // Bonus for reasonable position change
                }
            }
            
            return (candidate, score)
        }
        
        // Return candidate with highest score
        return scoredCandidates.max(by: { $0.1 < $1.1 })?.0
    }
    
    // Lightweight validation for immediate response
    private func validateAndReturnBestCandidate(_ candidate: BallDetectionResult?, currentTime: TimeInterval) -> BallDetectionResult? {
        guard let candidate = candidate else { return nil }
        
        // Add to history for velocity calculation
        detectionHistory.append(candidate)
        if detectionHistory.count > Config.maxTrackingHistory {
            detectionHistory.removeFirst()
        }
        
        // For fast balls, skip heavy filtering and return immediately
        // Update tracking state
        lastValidPosition = candidate.position
        currentVelocity = candidate.velocity
        
        // Add to confirmed detections for trajectory
        confirmedDetections.append(candidate)
        if confirmedDetections.count > Config.maxTrackingHistory {
            confirmedDetections.removeFirst()
        }
        
        print("[BallDetectionManager] ⚡ FAST TRACK: Immediate ball detection returned")
        return candidate
    }
    
    // Keep original temporal filtering for reference (unused for now)
    private func processTemporalFiltering(_ candidate: BallDetectionResult?, currentTime: TimeInterval) -> BallDetectionResult? {
        guard let candidate = candidate else { return nil }
        
        // Add to history
        detectionHistory.append(candidate)
        if detectionHistory.count > Config.maxTrackingHistory {
            detectionHistory.removeFirst()
        }
        
        // Apply temporal smoothing
        let smoothedCandidate = applySpatialSmoothing(candidate)
        
        // Require multiple consecutive detections for confirmation
        let recentDetections = detectionHistory.suffix(Config.minimumConfidenceFrames)
        guard recentDetections.count >= Config.minimumConfidenceFrames else {
            return nil  // Not enough history yet
        }
        
        // Validate motion consistency across recent frames
        guard isMotionConsistent(in: Array(recentDetections)) else {
            return nil  // Motion is too erratic
        }
        
        // Update confirmed detection
        confirmedDetections.append(smoothedCandidate)
        if confirmedDetections.count > Config.maxTrackingHistory {
            confirmedDetections.removeFirst()
        }
        
        // Update tracking state
        lastValidPosition = smoothedCandidate.position
        currentVelocity = smoothedCandidate.velocity
        
        return smoothedCandidate
    }
    
    private func applySpatialSmoothing(_ candidate: BallDetectionResult) -> BallDetectionResult {
        guard detectionHistory.count >= 2 else { return candidate }
        
        // Use weighted average of recent positions
        let recentDetections = detectionHistory.suffix(3)
        var weightedX: CGFloat = 0
        var weightedY: CGFloat = 0
        var totalWeight: CGFloat = 0
        
        for (index, detection) in recentDetections.enumerated() {
            let weight = CGFloat(index + 1)  // More recent = higher weight
            weightedX += detection.position.x * weight
            weightedY += detection.position.y * weight
            totalWeight += weight
        }
        
        let smoothedPosition = CGPoint(
            x: weightedX / totalWeight,
            y: weightedY / totalWeight
        )
        
        // Create smoothed result
        var smoothedBoundingBox = candidate.boundingBox
        smoothedBoundingBox.origin.x = smoothedPosition.x - smoothedBoundingBox.width / 2
        smoothedBoundingBox.origin.y = smoothedPosition.y - smoothedBoundingBox.height / 2
        
        return BallDetectionResult(
            position: smoothedPosition,
            boundingBox: smoothedBoundingBox,
            confidence: candidate.confidence,
            velocity: candidate.velocity,
            timestamp: candidate.timestamp
        )
    }
    
    private func isMotionConsistent(in detections: [BallDetectionResult]) -> Bool {
        guard detections.count >= 2 else { return true }
        
        var velocityChanges: [CGFloat] = []
        
        for i in 1..<detections.count {
            let prev = detections[i-1]
            let curr = detections[i]
            
            let velocityChange = sqrt(
                pow(curr.velocity.dx - prev.velocity.dx, 2) +
                pow(curr.velocity.dy - prev.velocity.dy, 2)
            )
            
            velocityChanges.append(velocityChange)
        }
        
        // Check if velocity changes are reasonable (not too erratic)
        let avgVelocityChange = velocityChanges.reduce(0, +) / CGFloat(velocityChanges.count)
        return avgVelocityChange < 0.5  // Threshold for erratic motion
    }
    
    private func processMotionBasedDetection(currentTime: TimeInterval) -> BallDetectionResult? {
        // TODO: Implement optical flow-based detection for when ML models fail
        // This would track movement patterns even without explicit ball detections
        return nil
    }
    
    // MARK: - Physics Validation
    
    /// Validate if detected motion follows basketball physics
    func validatePhysics(trajectory: [BallDetectionResult]) -> Bool {
        guard trajectory.count >= 3 else { return true }
        
        // Check for realistic basketball motion patterns
        var isValidPhysics = true
        
        // 1. Check for reasonable acceleration (gravity effect)
        for i in 2..<trajectory.count {
            let p1 = trajectory[i-2]
            let p2 = trajectory[i-1] 
            let p3 = trajectory[i]
            
            // Calculate acceleration in Y direction (should show gravity effect)
            let dt1 = p2.timestamp - p1.timestamp
            let dt2 = p3.timestamp - p2.timestamp
            
            if dt1 > 0 && dt2 > 0 {
                let v1y = (p2.position.y - p1.position.y) / CGFloat(dt1)
                let v2y = (p3.position.y - p2.position.y) / CGFloat(dt2)
                let acceleration = (v2y - v1y) / CGFloat(dt2)
                
                // Gravity should cause downward acceleration (positive Y is down)
                // But allow for upward motion during shot arc
                if abs(acceleration) > 5.0 {  // Unrealistic acceleration
                    isValidPhysics = false
                    break
                }
            }
        }
        
        return isValidPhysics
    }
}

// MARK: - Optical Flow Tracker (Placeholder)
private class OpticalFlowTracker {
    // TODO: Implement optical flow tracking for motion-based detection
    // This would use CoreImage or Vision framework to track movement patterns
    // even when explicit object detection fails
}