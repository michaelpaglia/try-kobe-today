import SwiftUI
import AVFoundation
import Vision
import Foundation
import CoreImage
import Accelerate
import CoreML
import Roboflow
// import Roboflow // Roboflow integration pending or not installed

// MARK: - Detection Structures
struct Detection: Equatable, Hashable {
    let boundingBox: CGRect
    let confidence: Float
    let label: String
    let isPerson: Bool
    let keypoints: [CGPoint]?  // For pose detection
    let trajectory: [CGPoint]? // For ball tracking
}

// MARK: - Shot State
enum ShotState {
    case none
    case preparation
    case release
    case followThrough
    case complete
}

// MARK: - Trajectory Analysis
struct TrajectoryPoint {
    let position: CGPoint
    let timestamp: TimeInterval
    let velocity: CGVector?
}

// Class names for YOLO output
let classNames = ["basketball", "rim", "person"]

// MARK: - Helper Functions
func scaledPoint(_ point: VNRecognizedPoint, in scale: CGSize) -> CGPoint {
    CGPoint(x: CGFloat(point.x) * scale.width, y: (1 - CGFloat(point.y)) * scale.height)
}

func angleDegrees(center: CGPoint, from: CGPoint, to: CGPoint) -> (start: CGFloat, end: CGFloat) {
    let startRadians = atan2(from.y - center.y, from.x - center.x)
    let endRadians = atan2(to.y - center.y, to.x - center.x)
    let startDegrees = startRadians * 180.0 / .pi
    let endDegrees = endRadians * 180.0 / .pi
    return (startDegrees, endDegrees)
}

// Preprocess pixel buffer: resize with letterbox to 640x640, keep aspect ratio, output BGRA
func preprocessPixelBuffer(_ pixelBuffer: CVPixelBuffer, targetSize: CGSize) -> CVPixelBuffer? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let inputSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
    let scale = min(targetSize.width / inputSize.width, targetSize.height / inputSize.height)
    let scaledSize = CGSize(width: inputSize.width * scale, height: inputSize.height * scale)
    let x = (targetSize.width - scaledSize.width) / 2
    let y = (targetSize.height - scaledSize.height) / 2
    let transform = CGAffineTransform(scaleX: scale, y: scale)
    let scaledImage = ciImage.transformed(by: transform)
    // Create a black background
    let background = CIImage(color: .black).cropped(to: CGRect(origin: .zero, size: targetSize))
    // Composite scaled image over black background (letterbox)
    let letterboxed = scaledImage.transformed(by: CGAffineTransform(translationX: x, y: y)).composited(over: background)
    // Create output buffer
    var outputBuffer: CVPixelBuffer?
    let attrs = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32BGRA)
    ] as CFDictionary
    CVPixelBufferCreate(kCFAllocatorDefault, Int(targetSize.width), Int(targetSize.height), kCVPixelFormatType_32BGRA, attrs, &outputBuffer)
    guard let outBuffer = outputBuffer else { return nil }
    let context = CIContext()
    context.render(letterboxed, to: outBuffer)
    return outBuffer
}

// Helper: Convert BGRA pixel buffer to RGB pixel buffer
func bgraToRgb(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    var rgbBuffer: CVPixelBuffer?
    let attrs = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferPixelFormatTypeKey: Int(kCVPixelFormatType_32RGBA)
    ] as CFDictionary
    CVPixelBufferCreate(kCFAllocatorDefault, CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer), kCVPixelFormatType_32RGBA, attrs, &rgbBuffer)
    guard let outBuffer = rgbBuffer else { return nil }
    context.render(ciImage, to: outBuffer)
    return outBuffer
}

// Convert BGRA pixel buffer to normalized (0-1) Float32 MLMultiArray in RGB order
func pixelBufferToNormalizedArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
    guard let array = try? MLMultiArray(shape: [3, NSNumber(value: height), NSNumber(value: width)], dataType: .float32) else { return nil }
    let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
    for y in 0..<height {
        for x in 0..<width {
            let offset = y * bytesPerRow + x * 4
            let b = Float32(buffer[offset + 0]) / 255.0
            let g = Float32(buffer[offset + 1]) / 255.0
            let r = Float32(buffer[offset + 2]) / 255.0
            // Write to MLMultiArray in RGB order
            ptr[0 * width * height + y * width + x] = r
            ptr[1 * width * height + y * width + x] = g
            ptr[2 * width * height + y * width + x] = b
        }
    }
    return array
}

// Helper: Convert CVPixelBuffer to UIImage
func pixelBufferToUIImage(_ pixelBuffer: CVPixelBuffer) -> UIImage? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
        return UIImage(cgImage: cgImage)
    }
    return nil
}

// Kalman Filter for smooth tracking
class KalmanFilter {
    private var x: Float = 0.0  // Position
    private var v: Float = 0.0  // Velocity
    private var p: Float = 1.0  // Error covariance
    private let q: Float = 0.1  // Process noise
    private let r: Float = 0.1  // Measurement noise
    
    func update(measurement: Float) -> Float {
        // Predict
        x = x + v
        p = p + q
        
        // Update
        let k = p / (p + r)  // Kalman gain
        x = x + k * (measurement - x)
        v = v + k * (measurement - x)
        p = (1 - k) * p
        
        return x
    }
}

// Trajectory prediction
func predictTrajectory(points: [TrajectoryPoint], numPoints: Int = 10) -> [CGPoint] {
    guard points.count >= 2 else { return [] }
    
    let velocities = zip(points, points.dropFirst()).map { p1, p2 in
        let dx = p2.position.x - p1.position.x
        let dy = p2.position.y - p1.position.y
        let dt = p2.timestamp - p1.timestamp
        return CGVector(dx: dx / CGFloat(dt), dy: dy / CGFloat(dt))
    }
    
    let avgVelocity = velocities.reduce(CGVector.zero) { v1, v2 in
        CGVector(dx: v1.dx + v2.dx, dy: v1.dy + v2.dy)
    }
    
    let lastPoint = points.last!
    var predictedPoints: [CGPoint] = []
    
    for i in 1...numPoints {
        let t = TimeInterval(i) * 0.1
        let x = lastPoint.position.x + avgVelocity.dx * CGFloat(t)
        let y = lastPoint.position.y + avgVelocity.dy * CGFloat(t) + 0.5 * 9.8 * CGFloat(t * t) // Add gravity
        predictedPoints.append(CGPoint(x: x, y: y))
    }
    
    return predictedPoints
}

// Helper to get the actual image size sent to Roboflow
fileprivate func getRoboflowInputSize(from image: UIImage) -> (CGFloat, CGFloat) {
    return (image.size.width, image.size.height)
}

// Preprocess image for Roboflow detection (resize to 640x640 with letterboxing)
fileprivate func preprocessImageForRoboflow(_ image: UIImage) -> UIImage {
    let targetSize = CGSize(width: 640, height: 640)
    
    // Calculate scaling to maintain aspect ratio
    let widthRatio = targetSize.width / image.size.width
    let heightRatio = targetSize.height / image.size.height
    let scale = min(widthRatio, heightRatio)
    
    let scaledSize = CGSize(width: image.size.width * scale, height: image.size.height * scale)
    let xOffset = (targetSize.width - scaledSize.width) / 2
    let yOffset = (targetSize.height - scaledSize.height) / 2
    
    UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
    
    // Fill with black background
    UIColor.black.setFill()
    UIRectFill(CGRect(origin: .zero, size: targetSize))
    
    // Draw scaled image centered
    image.draw(in: CGRect(x: xOffset, y: yOffset, width: scaledSize.width, height: scaledSize.height))
    
    let preprocessedImage = UIGraphicsGetImageFromCurrentImageContext() ?? image
    UIGraphicsEndImageContext()
    
    return preprocessedImage
}

// Helper to convert any number type to CGFloat
fileprivate func cgfloatFromAny(_ any: Any?) -> CGFloat {
    if let f = any as? Float { return CGFloat(f) }
    if let d = any as? Double { return CGFloat(d) }
    if let i = any as? Int { return CGFloat(i) }
    return 0
}

// Map Vision framework joint names to standard VNHumanBodyPoseObservation.JointName
fileprivate func mapVisionJoint(_ pointKey: VNRecognizedPointKey) -> VNHumanBodyPoseObservation.JointName? {
    // Convert pointKey to string to match against known patterns
    let keyString = "\(pointKey)"
    
    switch keyString {
    case let s where s.contains("head"):
        return .nose
    case let s where s.contains("left_eye"):
        return .leftEye
    case let s where s.contains("right_eye"):
        return .rightEye
    case let s where s.contains("left_ear"):
        return .leftEar
    case let s where s.contains("right_ear"):
        return .rightEar
    case let s where s.contains("left_shoulder"):
        return .leftShoulder
    case let s where s.contains("right_shoulder"):
        return .rightShoulder
    case let s where s.contains("left_forearm"):
        return .leftElbow
    case let s where s.contains("right_forearm"):
        return .rightElbow
    case let s where s.contains("left_hand"):
        return .leftWrist
    case let s where s.contains("right_hand"):
        return .rightWrist
    case let s where s.contains("left_upLeg"):
        return .leftHip
    case let s where s.contains("right_upLeg"):
        return .rightHip
    case let s where s.contains("left_leg"):
        return .leftKnee
    case let s where s.contains("right_leg"):
        return .rightKnee
    case let s where s.contains("left_foot"):
        return .leftAnkle
    case let s where s.contains("right_foot"):
        return .rightAnkle
    default:
        return nil
    }
}

// MARK: - CameraView
struct CameraView: UIViewControllerRepresentable {
    @Binding var posePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
    @Binding var poseObservation: VNHumanBodyPoseObservation?
    var useFrontCamera: Bool
    @Binding var segmentationMask: CGImage?
    @Binding var hoopBoundingBoxes: [Detection]
    @Binding var userDefinedHoopArea: CGRect?
    @Binding var shotDetected: Bool
    @Binding var personLocation: CGRect?

    func makeUIViewController(context: Context) -> CameraViewController {
        let controller = CameraViewController()
        controller.useFrontCamera = useFrontCamera
        controller.onPosePointsUpdate = { points, observation in
            DispatchQueue.main.async {
                self.posePoints = points
                self.poseObservation = observation
            }
        }
        controller.onSegmentationMaskUpdate = { mask in
            DispatchQueue.main.async {
                self.segmentationMask = mask
            }
        }
        controller.onBasketballDetectionsUpdate = { detections in
            DispatchQueue.main.async {
                self.hoopBoundingBoxes = detections
            }
        }
        controller.onPersonLocationUpdate = { location in
            DispatchQueue.main.async {
                self.personLocation = location
            }
        }
        controller.onShotDetected = { detected in
            DispatchQueue.main.async {
                self.shotDetected = detected
            }
        }
        return controller
    }

    func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {
        let cameraChanged = uiViewController.useFrontCamera != useFrontCamera
        uiViewController.useFrontCamera = useFrontCamera
        uiViewController.userDefinedHoopArea = userDefinedHoopArea
        if cameraChanged {
            uiViewController.clearDetections()
        }
        uiViewController.restartSessionIfNeeded()
    }
}

// MARK: - CameraViewController
class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let videoOutput = AVCaptureVideoDataOutput()
    private let visionQueue = DispatchQueue(label: "visionQueue")
    
    // Callbacks
    var onPosePointsUpdate: (([VNHumanBodyPoseObservation.JointName: VNRecognizedPoint], VNHumanBodyPoseObservation?) -> Void)?
    var onSegmentationMaskUpdate: ((CGImage?) -> Void)?
    var onBasketballDetectionsUpdate: (([Detection]) -> Void)?
    var onPersonLocationUpdate: ((CGRect?) -> Void)?
    var onShotDetected: ((Bool) -> Void)?
    
    // Camera settings
    var useFrontCamera: Bool = false
    var userDefinedHoopArea: CGRect? = nil
    private var currentCameraPosition: AVCaptureDevice.Position = .back
    
    // Detection and tracking
    private var yoloModel: MLModel?
    private var kalmanFilters: [String: (x: KalmanFilter, y: KalmanFilter)] = [:]
    private var trajectoryPoints: [String: [TrajectoryPoint]] = [:]
    private var currentShotState: ShotState = .none
    private var lastShotTime: TimeInterval = 0
    private var ballInHoopFrames: Int = 0
    private let requiredFramesInHoop: Int = 3
    private var lastBallPosition: CGRect?
    private var ballTrajectory: [CGPoint] = []
    private var activeDetections: [String: Detection] = [:] // Replaces bestDetections
    
    // Ball tracking and smoothing
    private var ballHistory: [Detection] = []
    private let maxBallHistory = 5
    private var ballVelocity: CGVector = .zero
    private var lastBallTime: TimeInterval = 0
    
    // Performance optimization
    private let processingQueue = DispatchQueue(label: "com.kobe.processing", qos: .userInteractive)
    private var lastProcessedFrame: TimeInterval = 0
    private let minimumFrameInterval: TimeInterval = 1.0 / 30.0 // 30 FPS target
    
    // Detection thresholds
    private let confidenceThreshold: Float = 0.3
    private let iouThreshold: Float = 0.5
    private let trajectoryHistorySize: Int = 10
    private let confidenceImprovementThreshold: Float = 0.05
    
    // Motion detection
    private var lastFrameAverageColor: (r: Float, g: Float, b: Float)?
    private let motionThreshold: Float = 0.02
    private var lastDetectionTime: Date = Date()
    private var lastCameraPosition: AVCaptureDevice.Position = .back
    
    private var rf: RoboflowMobile?
    private var model: RFObjectDetectionModel?
    private let roboflowApiKey = "HPniCnBD4zxfFXO1qz3A"
    private let roboflowModelId = "basketball-hoop-images"
    private let roboflowModelVersion = 2
    
    private var frameCounter: Int = 0
    private var visionRequest: VNDetectHumanBodyPoseRequest?
    private var lastPosePoints: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
    private var isProcessingSegments: Bool = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        print("CameraViewController loaded")
        setupCamera()
        // Setup Vision pose request
        visionRequest = VNDetectHumanBodyPoseRequest(completionHandler: { [weak self] request, error in
            guard let self = self else { return }
            if let results = request.results as? [VNHumanBodyPoseObservation], let first = results.first {
                var points: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint] = [:]
                // Get all available points and create proper mapping
                if let allPoints = try? first.recognizedPoints(forGroupKey: .all) {
                    for (pointKey, point) in allPoints {
                        if point.confidence > 0.3 {
                            // Map Vision framework joint names to standard VNHumanBodyPoseObservation.JointName
                            if let mappedJoint = mapVisionJoint(pointKey) {
                                points[mappedJoint] = point
                            }
                        }
                    }
                }
                let jointCount = points.count
                let confidences = points.mapValues { $0.confidence }
                print("Pose detected: \(jointCount) joints")
                
                // Debug: Print a few key joint positions to verify orientation
                if let nose = points[.nose] {
                    print("Head/Nose position: x=\(nose.x), y=\(nose.y)")
                }
                if let leftShoulder = points[.leftShoulder] {
                    print("Left shoulder position: x=\(leftShoulder.x), y=\(leftShoulder.y)")
                }
                if let rightShoulder = points[.rightShoulder] {
                    print("Right shoulder position: x=\(rightShoulder.x), y=\(rightShoulder.y)")
                }
                
                // Store for person bounding box creation
                self.lastPosePoints = points
                
                self.onPosePointsUpdate?(points, first)
            } else {
                print("No pose detected")
                self.onPosePointsUpdate?([:], nil)
            }
        })
        rf = RoboflowMobile(apiKey: roboflowApiKey)
        rf?.load(model: roboflowModelId, modelVersion: roboflowModelVersion) { [weak self] loadedModel, error, _, _ in
            if let loadedModel = loadedModel {
                // Very low thresholds to detect more objects
                loadedModel.configure(threshold: 0.05, overlap: 0.2, maxObjects: 20)
                self?.model = loadedModel
                print("Roboflow model loaded successfully with threshold: 0.05")
            } else if let error = error {
                print("Roboflow model failed to load: \(error.localizedDescription)")
            }
        }
    }
    
    private func setupCamera() {
        captureSession.beginConfiguration()
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video,
                                                       position: useFrontCamera ? .front : .back),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
              captureSession.canAddInput(videoInput) else {
            return
        }
        
        captureSession.addInput(videoInput)
        
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
            
            // Set video orientation for portrait mode
            if let connection = videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
        }
        
        captureSession.commitConfiguration()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer?.videoGravity = .resizeAspectFill
        previewLayer?.frame = view.bounds
        
        // Set preview layer orientation for portrait
        if let connection = previewLayer?.connection, connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
        
        view.layer.addSublayer(previewLayer!)
        
        captureSession.startRunning()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastProcessedFrame >= minimumFrameInterval else { return }
        lastProcessedFrame = currentTime
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        processingQueue.async { [weak self] in
            self?.processFrame(pixelBuffer)
        }
    }

    private func processFrame(_ pixelBuffer: CVPixelBuffer) {
        frameCounter += 1
        if frameCounter % 5 != 0 { return }
        print("processFrame called (frame: \(frameCounter))")
        
        // Check for camera movement and clear detections if significant motion detected
        if let currentColor = calculateAverageColor(pixelBuffer) {
            _ = hasSignificantMotion(currentColor)
            lastFrameAverageColor = currentColor
        }
        
        guard let model = self.model, let image = pixelBufferToUIImage(pixelBuffer) else {
            print("Model or image not available")
            return
        }
        // Run Vision pose detection with .up orientation for both cameras
        // Handle front camera mirroring in coordinate transformation
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        if let visionRequest = visionRequest {
            do {
                try handler.perform([visionRequest])
            } catch {
                print("Vision pose detection error: \(error)")
            }
        }
        // Segment image into 640x640 squares and process each (throttled)
        guard !isProcessingSegments else {
            print("Still processing previous segments, skipping frame")
            return
        }
        
        isProcessingSegments = true
        processImageSegments(image, model: model) { [weak self] allDetections in
            guard let self = self else { return }
            print("Total detections from all segments: \(allDetections.count)")
            self.updateDetections(allDetections)
            self.isProcessingSegments = false
        }
    }

    // Validate basketball size relative to person
    private func isValidBallSize(_ ballBox: CGRect, personBoundingBox: CGRect?) -> Bool {
        guard let personBox = personBoundingBox else {
            // Without person reference, just check basic size constraints
            let area = ballBox.width * ballBox.height
            return area >= 0.005 && area <= 0.1 // Between 0.5% and 10% of screen
        }
        
        let ballArea = ballBox.width * ballBox.height
        let personArea = personBox.width * personBox.height
        
        // Basketball should be between 1/50 and 1/8 of person area
        let minBallRatio: CGFloat = 1.0 / 50.0
        let maxBallRatio: CGFloat = 1.0 / 8.0
        let ballRatio = ballArea / personArea
        
        // Check aspect ratio - basketball should be roughly circular
        let aspectRatio = ballBox.width / ballBox.height
        let isCircular = aspectRatio >= 0.6 && aspectRatio <= 1.4
        
        // Check minimum absolute size to avoid tiny false positives
        let minAbsoluteArea: CGFloat = 0.002 // 0.2% of screen
        
        let isValidSize = ballRatio >= minBallRatio && ballRatio <= maxBallRatio
        let hasMinSize = ballArea >= minAbsoluteArea
        
        if !isValidSize || !isCircular || !hasMinSize {
            print("Ball validation failed - ratio: \(ballRatio), circular: \(isCircular), minSize: \(hasMinSize)")
        }
        
        return isValidSize && isCircular && hasMinSize
    }
    
    // Smooth ball detection using temporal tracking
    private func smoothBallDetection(_ ballDetection: Detection) -> Detection {
        let currentTime = CACurrentMediaTime()
        
        // Add to history
        ballHistory.append(ballDetection)
        if ballHistory.count > maxBallHistory {
            ballHistory.removeFirst()
        }
        
        // Calculate velocity if we have previous detection
        if let lastBall = ballHistory.dropLast().last, lastBallTime > 0 {
            let deltaTime = currentTime - lastBallTime
            if deltaTime > 0 {
                let deltaX = ballDetection.boundingBox.midX - lastBall.boundingBox.midX
                let deltaY = ballDetection.boundingBox.midY - lastBall.boundingBox.midY
                ballVelocity = CGVector(dx: deltaX / deltaTime, dy: deltaY / deltaTime)
            }
        }
        
        lastBallTime = currentTime
        
        // If we have enough history, smooth the position
        guard ballHistory.count >= 3 else { return ballDetection }
        
        // Weighted average of recent detections (more recent = higher weight)
        var weightedX: CGFloat = 0
        var weightedY: CGFloat = 0
        var weightedWidth: CGFloat = 0
        var weightedHeight: CGFloat = 0
        var totalWeight: CGFloat = 0
        
        for (index, detection) in ballHistory.enumerated() {
            let weight = CGFloat(index + 1) // Linear weight increase
            weightedX += detection.boundingBox.midX * weight
            weightedY += detection.boundingBox.midY * weight
            weightedWidth += detection.boundingBox.width * weight
            weightedHeight += detection.boundingBox.height * weight
            totalWeight += weight
        }
        
        let smoothedCenter = CGPoint(x: weightedX / totalWeight, y: weightedY / totalWeight)
        let smoothedSize = CGSize(width: weightedWidth / totalWeight, height: weightedHeight / totalWeight)
        
        let smoothedBox = CGRect(
            x: smoothedCenter.x - smoothedSize.width / 2,
            y: smoothedCenter.y - smoothedSize.height / 2,
            width: smoothedSize.width,
            height: smoothedSize.height
        )
        
        return Detection(
            boundingBox: smoothedBox,
            confidence: ballDetection.confidence, // Keep original confidence
            label: ballDetection.label,
            isPerson: false,
            keypoints: nil,
            trajectory: nil
        )
    }
    
    // Check if ball is near person's hands for shot detection
    private func isNearHands(_ ballBox: CGRect) -> Bool {
        guard !lastPosePoints.isEmpty else { return false }
        
        let ballCenter = CGPoint(x: ballBox.midX, y: ballBox.midY)
        let proximityThreshold: CGFloat = 0.1 // 10% of screen distance
        
        // Check distance to left and right wrists
        let handsToCheck: [VNHumanBodyPoseObservation.JointName] = [.leftWrist, .rightWrist]
        
        for hand in handsToCheck {
            if let handPoint = lastPosePoints[hand] {
                // Apply same coordinate transformations as in pose overlay
                let handX = useFrontCamera ? (1.0 - handPoint.x) : handPoint.x
                let handY = 1.0 - handPoint.y
                let handPosition = CGPoint(x: handX, y: handY)
                
                let distance = sqrt(pow(ballCenter.x - handPosition.x, 2) + pow(ballCenter.y - handPosition.y, 2))
                if distance <= proximityThreshold && handPoint.confidence > 0.5 {
                    print("Ball near \(hand): distance=\(distance)")
                    return true
                }
            }
        }
        
        return false
    }

    private func updateDetections(_ newDetections: [Detection]) {
        print("updateDetections called with: \(newDetections)")
        var updated = false
        var personDetection: Detection?
        var validBallDetections: [Detection] = []
        
        // First pass: collect person detection and filter ball detections by size
        for detection in newDetections {
            if detection.label == "person" {
                personDetection = detection
                onPersonLocationUpdate?(detection.boundingBox)
            } else if detection.label == "basketball" {
                // Get person bounding box for size comparison
                let personBoundingBox = personDetection?.boundingBox ?? createPersonBoundingBoxFromPose()
                if isValidBallSize(detection.boundingBox, personBoundingBox: personBoundingBox) {
                    validBallDetections.append(detection)
                } else {
                    print("Filtering out basketball with invalid size: \(detection.boundingBox)")
                    continue
                }
            }
        }
        
        // Select best ball detection from valid ones and apply temporal smoothing
        var ballDetection: Detection? = nil
        if let bestBall = validBallDetections.max(by: { $0.confidence < $1.confidence }) {
            ballDetection = smoothBallDetection(bestBall)
            
            // Check if ball is near hands for enhanced shot detection
            if isNearHands(ballDetection!.boundingBox) {
                print("Ball detected near hands - potential shot setup")
            }
        }
        
        // Process all detections (including smoothed ball detection)
        var detectionsToProcess = newDetections.filter { $0.label != "basketball" }
        if let validBall = ballDetection {
            detectionsToProcess.append(validBall)
        }
        
        for detection in detectionsToProcess {
            // Lower threshold for basketball/rim vs general objects  
            let threshold = (detection.label == "basketball" || detection.label.lowercased().contains("rim") || detection.label.lowercased().contains("hoop")) ? 0.1 : confidenceThreshold
            guard detection.confidence >= threshold else { 
                print("Filtering out \(detection.label) with confidence \(detection.confidence) (threshold: \(threshold))")
                continue 
            }
            
            if let existingDetection = activeDetections[detection.label] {
                if detection.confidence > existingDetection.confidence + confidenceImprovementThreshold {
                    activeDetections[detection.label] = detection
                    updated = true
                }
            } else {
                activeDetections[detection.label] = detection
                updated = true
            }
        }
        
        // Check for shots if we have both ball and person detections
        if let ball = ballDetection {
            // Use person detection from Roboflow, or create one from pose data
            let personBoundingBox = personDetection?.boundingBox ?? createPersonBoundingBoxFromPose()
            detectShot(ballDetection: ball, personLocation: personBoundingBox)
            
            // Update person location for UI
            if personBoundingBox != nil {
                onPersonLocationUpdate?(personBoundingBox)
            }
        }
        
        // If we have updates, send them to the UI
        if updated {
            DispatchQueue.main.async {
                self.onBasketballDetectionsUpdate?(Array(self.activeDetections.values))
            }
        }
        
        // Update ball trajectory
        if let ball = ballDetection {
            let ballCenter = CGPoint(x: ball.boundingBox.midX, y: ball.boundingBox.midY)
            ballTrajectory.append(ballCenter)
            if ballTrajectory.count > trajectoryHistorySize {
                ballTrajectory.removeFirst()
            }
        }
    }

    private func detectShot(ballDetection: Detection, personLocation: CGRect?) {
        guard let personLocation = personLocation else { return }
        let ballLeftPerson = !ballDetection.boundingBox.intersects(personLocation)
        // Find rim/hoop detection with flexible naming
        let rimDetection = activeDetections.values.first { detection in
            let label = detection.label.lowercased()
            return label.contains("rim") || label.contains("hoop") || label.contains("basket")
        }
        
        let ballCenter = CGPoint(x: ballDetection.boundingBox.midX, y: ballDetection.boundingBox.midY)
        var shotInHoop = false
        
        // Check rim detection first
        if let rimDetection = rimDetection, ballLeftPerson {
            shotInHoop = rimDetection.boundingBox.contains(ballCenter)
        }
        // Fallback to user-defined hoop area
        else if let userHoopArea = userDefinedHoopArea, ballLeftPerson {
            shotInHoop = userHoopArea.contains(ballCenter)
        }
        
        if shotInHoop {
            ballInHoopFrames += 1
            if ballInHoopFrames >= requiredFramesInHoop {
                onShotDetected?(true)
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    self.ballInHoopFrames = 0
                    self.onShotDetected?(false)
                }
            }
        } else {
            ballInHoopFrames = 0
        }
        lastBallPosition = ballDetection.boundingBox
    }

    func clearDetections() {
        activeDetections.removeAll()
        ballTrajectory.removeAll()
        lastBallPosition = nil
        ballInHoopFrames = 0
        currentShotState = .none
        
        // Clear ball tracking data
        ballHistory.removeAll()
        ballVelocity = .zero
        lastBallTime = 0
        
        // Clear UI immediately
        DispatchQueue.main.async {
            self.onBasketballDetectionsUpdate?([])
            self.onPersonLocationUpdate?(nil)
            self.onShotDetected?(false)
        }
    }

    func restartSessionIfNeeded() {
        let desiredPosition: AVCaptureDevice.Position = useFrontCamera ? .front : .back
        if desiredPosition != currentCameraPosition {
            // Clear existing detections and reset all state
            activeDetections.removeAll()
            ballTrajectory.removeAll()
            kalmanFilters.removeAll()
            trajectoryPoints.removeAll()
            lastFrameAverageColor = nil
            lastDetectionTime = Date()
            lastCameraPosition = currentCameraPosition
            lastBallPosition = nil
            ballInHoopFrames = 0
            currentShotState = .none
            frameCounter = 0
            
            // Clear UI immediately
            DispatchQueue.main.async {
                self.onBasketballDetectionsUpdate?([])
                self.onPosePointsUpdate?([:], nil)
                self.onSegmentationMaskUpdate?(nil)
                self.onPersonLocationUpdate?(nil)
                self.onShotDetected?(false)
            }
            
            // Stop and reconfigure camera
            captureSession.stopRunning()
            for input in captureSession.inputs {
                captureSession.removeInput(input)
            }
            for output in captureSession.outputs {
                captureSession.removeOutput(output)
            }
            
            currentCameraPosition = desiredPosition
            setupCamera()
            
            // Force a detection update after camera switch
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                self.activeDetections.removeAll()
                self.onBasketballDetectionsUpdate?([])
            }
            
            // Restart session
            DispatchQueue.global(qos: .userInitiated).async {
                self.captureSession.startRunning()
            }
        }
    }

    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer)), y: CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))))
        var outputBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &outputBuffer)
        if let outputBuffer = outputBuffer {
            context.render(scaled, to: outputBuffer)
            return outputBuffer
        }
        return nil
    }

    private func decodeYOLOOutput(_ output: MLMultiArray, inputWidth: Int, inputHeight: Int, confidenceThreshold: Float = 0.01) -> [Detection] {
        var results: [Detection] = []
        let numAnchors = 8400
        let numAttrs = 6
        let rimThreshold: Float = 0.01
        let basketballThreshold: Float = 0.01
        guard output.shape.count == 3 else { return results }
        let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(output.dataPointer))
        for i in 0..<numAnchors {
            let base = i * numAttrs
            let x = ptr[base + 0]
            let y = ptr[base + 1]
            let w = ptr[base + 2]
            let h = ptr[base + 3]
            let conf = ptr[base + 4]
            let cls = ptr[base + 5]
            let classIndex = Int(round(cls))
            if classIndex >= 0 && classIndex < classNames.count {
                let label = classNames[classIndex]
                let threshold = (label == "rim") ? rimThreshold : basketballThreshold
                if conf > threshold {
                    let rect = CGRect(
                        x: CGFloat(x - w/2) / CGFloat(inputWidth),
                        y: CGFloat(y - h/2) / CGFloat(inputHeight),
                        width: CGFloat(w) / CGFloat(inputWidth),
                        height: CGFloat(h) / CGFloat(inputHeight)
                    )
                    let normRect = rect.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
                    if normRect.width > 0.02 && normRect.height > 0.02 {
                        let detection = Detection(
                            boundingBox: normRect,
                            confidence: conf,
                            label: label,
                            isPerson: label == "person",
                            keypoints: nil,
                            trajectory: nil
                        )
                        print("[CoreML] Detection label: \(label), bbox: \(normRect), confidence: \(conf)")
                        results.append(detection)
                    }
                }
            }
        }
        return results
    }

    // Add this helper function for motion detection
    private func calculateAverageColor(_ pixelBuffer: CVPixelBuffer) -> (r: Float, g: Float, b: Float)? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        var totalR: Float = 0
        var totalG: Float = 0
        var totalB: Float = 0
        _ = Float(width * height)
        
        // Sample every 10th pixel for performance
        for y in stride(from: 0, to: height, by: 10) {
            for x in stride(from: 0, to: width, by: 10) {
                let offset = y * bytesPerRow + x * 4
                totalB += Float(buffer[offset + 0]) / 255.0
                totalG += Float(buffer[offset + 1]) / 255.0
                totalR += Float(buffer[offset + 2]) / 255.0
            }
        }
        
        let sampleCount = Float((width/10) * (height/10))
        return (
            r: totalR / sampleCount,
            g: totalG / sampleCount,
            b: totalB / sampleCount
        )
    }
    
    private func hasSignificantMotion(_ currentColor: (r: Float, g: Float, b: Float)) -> Bool {
        guard let lastColor = lastFrameAverageColor else { return true }
        
        let colorDiff = sqrt(
            pow(currentColor.r - lastColor.r, 2) +
            pow(currentColor.g - lastColor.g, 2) +
            pow(currentColor.b - lastColor.b, 2)
        )
        
        // Clear detections if motion is significant or camera changed
        if colorDiff > motionThreshold || currentCameraPosition != lastCameraPosition {
            activeDetections.removeAll()
            DispatchQueue.main.async {
                self.onBasketballDetectionsUpdate?([])
            }
            lastCameraPosition = currentCameraPosition
        }
        
        return colorDiff > motionThreshold
    }
    
    // Create person bounding box from pose joints
    private func createPersonBoundingBoxFromPose() -> CGRect? {
        guard !lastPosePoints.isEmpty else { return nil }
        
        var minX: CGFloat = 1.0
        var maxX: CGFloat = 0.0
        var minY: CGFloat = 1.0
        var maxY: CGFloat = 0.0
        
        // Find bounding box of all detected joints, applying front camera mirroring and Y-flip if needed
        for (_, point) in lastPosePoints {
            let x = useFrontCamera ? (1.0 - point.x) : point.x
            let y = 1.0 - point.y  // Flip Y coordinate to match SwiftUI coordinate system
            
            minX = min(minX, x)
            maxX = max(maxX, x)
            minY = min(minY, y)
            maxY = max(maxY, y)
        }
        
        // Add some padding and ensure reasonable human proportions
        let padding: CGFloat = 0.05
        let detectedWidth = (maxX - minX) + padding
        let detectedHeight = (maxY - minY) + padding
        
        // Ensure minimum size and human-like proportions (taller than wide)
        let width = max(0.15, detectedWidth)
        let height = max(0.4, max(detectedHeight, width * 2.0)) // Height should be at least 2x width
        
        let centerX = (minX + maxX) / 2
        let centerY = (minY + maxY) / 2
        
        let boundingBox = CGRect(
            x: centerX - width/2,
            y: centerY - height/2,
            width: width,
            height: height
        )
        
        print("Created person bounding box from pose: \(boundingBox)")
        return boundingBox
    }
    
    // Process image in 640x640 segments for better detection
    private func processImageSegments(_ image: UIImage, model: RFObjectDetectionModel, completion: @escaping ([Detection]) -> Void) {
        let imageSize = image.size
        let segmentSize: CGFloat = 640
        
        // Calculate how many segments we can fit
        let cols = Int(ceil(imageSize.width / segmentSize))
        let rows = Int(ceil(imageSize.height / segmentSize))
        
        print("Processing image \(imageSize.width)x\(imageSize.height) in \(cols)x\(rows) segments")
        
        var allDetections: [Detection] = []
        let dispatchGroup = DispatchGroup()
        let detectionQueue = DispatchQueue(label: "detection-queue", attributes: .concurrent)
        
        for row in 0..<rows {
            for col in 0..<cols {
                dispatchGroup.enter()
                
                detectionQueue.async {
                    let x = CGFloat(col) * segmentSize
                    let y = CGFloat(row) * segmentSize
                    
                    // Create segment rect, ensuring we don't exceed image bounds
                    let segmentRect = CGRect(
                        x: x,
                        y: y,
                        width: min(segmentSize, imageSize.width - x),
                        height: min(segmentSize, imageSize.height - y)
                    )
                    
                    if let segmentImage = self.cropImage(image, to: segmentRect) {
                        let paddedSegment = self.padImageTo640x640(segmentImage)
                        
                        // Process this segment
                        model.detect(image: paddedSegment) { predictions, error in
                            defer { dispatchGroup.leave() }
                            
                            if let error = error {
                                print("Segment (\(row),\(col)) detection error: \(error)")
                                return
                            }
                            
                            guard let predictions = predictions else {
                                print("Segment (\(row),\(col)): No predictions")
                                return
                            }
                            
                            print("Segment (\(row),\(col)): \(predictions.count) predictions")
                            
                            let segmentDetections = predictions.compactMap { pred -> Detection? in
                                let values = pred.getValues()
                                let x = cgfloatFromAny(values["x"])
                                let y = cgfloatFromAny(values["y"])
                                let width = cgfloatFromAny(values["width"])
                                let height = cgfloatFromAny(values["height"])
                                let className = values["class"] as? String ?? ""
                                
                                let confidenceAny = values["confidence"]
                                let confidence: Float
                                if let c = confidenceAny as? Float {
                                    confidence = c
                                } else if let c = confidenceAny as? Double {
                                    confidence = Float(c)
                                } else {
                                    confidence = 0
                                }
                                
                                // Skip low confidence detections
                                guard confidence > 0.1 else { return nil }
                                
                                // Convert coordinates from segment space back to full image space
                                let fullImageX = (x / 640.0 * segmentRect.width + segmentRect.minX) / imageSize.width
                                let fullImageY = (y / 640.0 * segmentRect.height + segmentRect.minY) / imageSize.height
                                let fullImageWidth = (width / 640.0 * segmentRect.width) / imageSize.width
                                let fullImageHeight = (height / 640.0 * segmentRect.height) / imageSize.height
                                
                                let bbox = CGRect(
                                    x: fullImageX,
                                    y: fullImageY,
                                    width: fullImageWidth,
                                    height: fullImageHeight
                                )
                                
                                print("Segment (\(row),\(col)) detection: \(className) conf=\(confidence) bbox=\(bbox)")
                                
                                return Detection(
                                    boundingBox: bbox,
                                    confidence: confidence,
                                    label: className,
                                    isPerson: className == "person",
                                    keypoints: nil,
                                    trajectory: nil
                                )
                            }
                            
                            DispatchQueue.main.async {
                                allDetections.append(contentsOf: segmentDetections)
                            }
                        }
                    } else {
                        dispatchGroup.leave()
                    }
                }
            }
        }
        
        dispatchGroup.notify(queue: .main) {
            print("All segments processed. Total detections: \(allDetections.count)")
            completion(allDetections)
        }
    }
    
    // Crop image to specified rectangle
    private func cropImage(_ image: UIImage, to rect: CGRect) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        let croppedCGImage = cgImage.cropping(to: rect)
        return croppedCGImage.map { UIImage(cgImage: $0) }
    }
    
    // Pad image to 640x640 (center with black borders if needed)
    private func padImageTo640x640(_ image: UIImage) -> UIImage {
        let targetSize = CGSize(width: 640, height: 640)
        let imageSize = image.size
        
        // If already 640x640, return as-is
        if imageSize.width == 640 && imageSize.height == 640 {
            return image
        }
        
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        
        // Fill with black background
        UIColor.black.setFill()
        UIRectFill(CGRect(origin: .zero, size: targetSize))
        
        // Calculate centering position
        let x = (640 - imageSize.width) / 2
        let y = (640 - imageSize.height) / 2
        
        // Draw image centered
        image.draw(in: CGRect(x: x, y: y, width: imageSize.width, height: imageSize.height))
        
        let paddedImage = UIGraphicsGetImageFromCurrentImageContext() ?? image
        UIGraphicsEndImageContext()
        
        return paddedImage
    }
}

struct SegmentationOverlayView: View {
    let mask: CGImage?
    var body: some View {
        if let mask = mask {
            Image(decorative: mask, scale: 1.0, orientation: .up)
                .resizable()
                .scaledToFill()
                .opacity(0.5)
                .blendMode(.plusLighter)
        }
    }
}

// Add helper to get corners of a bounding box (for convex hull placeholder)
extension Detection {
    var boundingBoxCorners: [CGPoint] {
        let minX = boundingBox.minX
        let minY = boundingBox.minY
        let maxX = boundingBox.maxX
        let maxY = boundingBox.maxY
        return [
            CGPoint(x: minX, y: minY),
            CGPoint(x: maxX, y: minY),
            CGPoint(x: maxX, y: maxY),
            CGPoint(x: minX, y: maxY)
        ]
    }
}

// Add debug prints for pose detection (wherever pose detection is run)
// For example, inside your Vision pose detection callback:
// print("Pose detected: \(poseObservation)") 