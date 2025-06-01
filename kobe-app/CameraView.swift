import SwiftUI
import AVFoundation
import Vision

struct CameraView: UIViewControllerRepresentable {
    @Binding var posePoints: [CGPoint]
    var useFrontCamera: Bool
    @Binding var segmentationMask: CGImage?
    
    func makeUIViewController(context: Context) -> CameraViewController {
        let controller = CameraViewController()
        controller.useFrontCamera = useFrontCamera
        controller.onPosePointsUpdate = { points in
            DispatchQueue.main.async {
                self.posePoints = points
            }
        }
        controller.onSegmentationMaskUpdate = { mask in
            DispatchQueue.main.async {
                self.segmentationMask = mask
            }
        }
        return controller
    }
    
    func updateUIViewController(_ uiViewController: CameraViewController, context: Context) {
        uiViewController.useFrontCamera = useFrontCamera
        uiViewController.restartSessionIfNeeded()
    }
}

class CameraViewController: UIViewController {
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let videoOutput = AVCaptureVideoDataOutput()
    private let visionQueue = DispatchQueue(label: "visionQueue")
    var onPosePointsUpdate: (([CGPoint]) -> Void)?
    var onSegmentationMaskUpdate: ((CGImage?) -> Void)?
    private var lastPosePoints: [CGPoint] = []
    var useFrontCamera: Bool = false
    private var currentCameraPosition: AVCaptureDevice.Position = .back
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }
    
    func restartSessionIfNeeded() {
        let desiredPosition: AVCaptureDevice.Position = useFrontCamera ? .front : .back
        if desiredPosition != currentCameraPosition {
            captureSession.stopRunning()
            for input in captureSession.inputs {
                captureSession.removeInput(input)
            }
            currentCameraPosition = desiredPosition
            setupCamera()
        }
    }
    
    private func setupCamera() {
        let position: AVCaptureDevice.Position = useFrontCamera ? .front : .back
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let videoInput = try? AVCaptureDeviceInput(device: videoDevice),
              captureSession.canAddInput(videoInput) else {
            return
        }
        captureSession.addInput(videoInput)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
            videoOutput.setSampleBufferDelegate(self, queue: visionQueue)
        }
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        self.previewLayer = previewLayer
        
        captureSession.startRunning()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let poseRequest = VNDetectHumanBodyPoseRequest { [weak self] request, error in
            guard let self = self else { return }
            if let results = request.results as? [VNHumanBodyPoseObservation], let first = results.first {
                if let recognizedPoints = try? first.recognizedPoints(.all) {
                    let imageSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
                    let points = recognizedPoints.values.compactMap { point -> CGPoint? in
                        guard point.confidence > 0.3 else { return nil }
                        // Vision uses a normalized coordinate system (0,0) bottom-left, (1,1) top-right
                        return CGPoint(x: point.x * imageSize.width, y: (1 - point.y) * imageSize.height)
                    }
                    self.lastPosePoints = points
                    self.onPosePointsUpdate?(points)
                }
            }
        }
        let segmentationRequest = VNGeneratePersonSegmentationRequest()
        segmentationRequest.qualityLevel = .accurate
        segmentationRequest.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform([poseRequest, segmentationRequest])
        if let maskPixelBuffer = segmentationRequest.results?.first?.pixelBuffer {
            let ciImage = CIImage(cvPixelBuffer: maskPixelBuffer)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                self.onSegmentationMaskUpdate?(cgImage)
            }
        } else {
            self.onSegmentationMaskUpdate?(nil)
        }
    }
}

struct PoseOverlayView: View {
    let points: [CGPoint]
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Draw outline (convex hull) if enough points
                if points.count > 2 {
                    Path { path in
                        let scaledPoints = points.map { CGPoint(x: $0.x * geometry.size.width / 720, y: $0.y * geometry.size.height / 1280) }
                        if let first = scaledPoints.first {
                            path.move(to: first)
                            for pt in scaledPoints.dropFirst() {
                                path.addLine(to: pt)
                            }
                            path.addLine(to: first)
                        }
                    }
                    .stroke(Color.blue.opacity(0.6), lineWidth: 2)
                }
                // Draw keypoints
                ForEach(0..<points.count, id: \ .self) { i in
                    let point = points[i]
                    Circle()
                        .fill(Color.green)
                        .frame(width: 10, height: 10)
                        .position(x: point.x * geometry.size.width / 720, y: point.y * geometry.size.height / 1280)
                }
            }
        }
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