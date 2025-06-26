import SwiftUI
import Vision
import CoreGraphics

struct PoseOverlayView: View {
    let points: [VNHumanBodyPoseObservation.JointName: VNRecognizedPoint]
    let observation: VNHumanBodyPoseObservation?
    let currentAngle: CGFloat?
    let hoopBoundingBoxes: [Detection]
    let personLocation: CGRect?
    let shotDetected: Bool
    let userDefinedHoopArea: CGRect?
    let ballTrajectory: [CGPoint]
    let ballBox: CGRect?
    let useFrontCamera: Bool
    let currentShotPhase: String
    let shootingHand: String
    // Add all detections for tap-to-detect
    var allDetections: [Detection] {
        hoopBoundingBoxes // You can expand this if you have more types
    }
    // Tap-to-detect state
    @State private var selectedDetection: Detection? = nil

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                personBox(in: geometry)
                hoopBoxes(in: geometry)
                posePoints(in: geometry)
                angleView(in: geometry)
                shotIndicator(in: geometry)
                userDefinedHoopAreaView(in: geometry)
                ballBoxView(in: geometry)
                ballTrajectoryView(in: geometry)
                calculateAngles(in: geometry)
                debugInfo(in: geometry)
                segmentGrid(in: geometry)
                shotPhaseDisplay(in: geometry)
                // Draw CLEARLY LABELED detection boxes
                ForEach(hoopBoundingBoxes, id: \ .self) { detection in
                    if detection.label == "basketball" {
                        // MINI BALL DETECTION - Smooth responsive circle with convex hull effect
                        ZStack {
                            // Outer glow effect
                            Circle()
                                .fill(
                                    RadialGradient(
                                        gradient: Gradient(colors: [Color.green.opacity(0.6), Color.green.opacity(0.1), Color.clear]),
                                        center: .center,
                                        startRadius: 5,
                                        endRadius: 25
                                    )
                                )
                                .frame(width: 50, height: 50)
                            
                            // Main ball circle
                            Circle()
                                .stroke(Color.green, lineWidth: 3)
                                .background(Circle().fill(Color.green.opacity(0.3)))
                                .frame(width: 30, height: 30)
                            
                            // Center dot
                            Circle()
                                .fill(Color.green)
                                .frame(width: 8, height: 8)
                        }
                        .position(x: (detection.boundingBox.minX + detection.boundingBox.width/2) * geometry.size.width, y: (detection.boundingBox.minY + detection.boundingBox.height/2) * geometry.size.height)
                        .animation(.easeInOut(duration: 0.1), value: detection.boundingBox)
                    } else if detection.label.lowercased().contains("rim") || detection.label.lowercased().contains("hoop") {
                        // RIM/HOOP DETECTION - Bright orange rectangle
                        ZStack {
                            Rectangle()
                                .stroke(Color.orange, lineWidth: 4)
                                .background(Rectangle().fill(Color.orange.opacity(0.2)))
                                .frame(width: detection.boundingBox.width * geometry.size.width, height: detection.boundingBox.height * geometry.size.height)
                            
                            Text("🎯 RIM/HOOP")
                                .font(.caption.bold())
                                .foregroundColor(.orange)
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(4)
                        }
                        .position(x: (detection.boundingBox.minX + detection.boundingBox.width/2) * geometry.size.width, y: (detection.boundingBox.minY + detection.boundingBox.height/2) * geometry.size.height)
                    }
                }
                // Highlight selected detection with a thick border
                if let selected = selectedDetection {
                    Rectangle()
                        .stroke(Color.yellow, lineWidth: 6)
                        .frame(width: selected.boundingBox.width * geometry.size.width, height: selected.boundingBox.height * geometry.size.height)
                        .position(x: (selected.boundingBox.minX + selected.boundingBox.width/2) * geometry.size.width, y: (selected.boundingBox.minY + selected.boundingBox.height/2) * geometry.size.height)
                }
            }
            // Tap gesture for tap-to-detect
            .contentShape(Rectangle())
            .gesture(TapGesture().onEnded { tap in
                // Convert tap location to normalized coordinates
                let tapLocation = tapLocationInGeometryProxy(geometry)
                let normX = tapLocation.x / geometry.size.width
                let normY = tapLocation.y / geometry.size.height
                // Find detection containing tap
                if let found = allDetections.first(where: { $0.boundingBox.contains(CGPoint(x: normX, y: normY)) }) {
                    selectedDetection = found
                } else {
                    selectedDetection = nil
                }
            })
        }
    }
    // Helper to get tap location (for SwiftUI 2/3 compatibility)
    private func tapLocationInGeometryProxy(_ geometry: GeometryProxy) -> CGPoint {
        // This is a placeholder; in a real app, use a DragGesture or a custom UIViewRepresentable to get tap location
        // For now, return the center
        return CGPoint(x: geometry.size.width/2, y: geometry.size.height/2)
    }
    
    @ViewBuilder
    private func personBox(in geometry: GeometryProxy) -> some View {
        if let personRect = personLocation {
            Rectangle()
                .stroke(Color.green, lineWidth: 2)
                .frame(
                    width: personRect.width * geometry.size.width,
                    height: personRect.height * geometry.size.height
                )
                .position(
                    x: (personRect.minX + personRect.width/2) * geometry.size.width,
                    y: (personRect.minY + personRect.height/2) * geometry.size.height
                )
        }
    }
    
    @ViewBuilder
    private func hoopBoxes(in geometry: GeometryProxy) -> some View {
        if let userDefinedHoopArea = userDefinedHoopArea {
            Rectangle()
                .stroke(Color.yellow, lineWidth: 2)
                .frame(
                    width: userDefinedHoopArea.width * geometry.size.width,
                    height: userDefinedHoopArea.height * geometry.size.height
                )
                .position(
                    x: (userDefinedHoopArea.minX + userDefinedHoopArea.width/2) * geometry.size.width,
                    y: (userDefinedHoopArea.minY + userDefinedHoopArea.height/2) * geometry.size.height
                )
        }
        
        if userDefinedHoopArea == nil {
            ForEach(hoopBoundingBoxes, id: \.self) { detection in
                HoopBoxView(detection: detection, geometry: geometry)
            }
        }
    }
    
    private struct HoopBoxView: View {
        let detection: Detection
        let geometry: GeometryProxy
        var body: some View {
            Rectangle()
                .stroke(Color.red, lineWidth: 2)
                .frame(
                    width: detection.boundingBox.width * geometry.size.width,
                    height: detection.boundingBox.height * geometry.size.height
                )
                .position(
                    x: (detection.boundingBox.minX + detection.boundingBox.width/2) * geometry.size.width,
                    y: (detection.boundingBox.minY + detection.boundingBox.height/2) * geometry.size.height
                )
        }
    }
    
    @ViewBuilder
    private func posePoints(in geometry: GeometryProxy) -> some View {
        // Draw skeleton lines using Apple Vision joint names
        let jointPairs: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
            // Arms
            (.leftShoulder, .leftElbow),
            (.leftElbow, .leftWrist),
            (.rightShoulder, .rightElbow),
            (.rightElbow, .rightWrist),
            // Torso
            (.leftShoulder, .rightShoulder),
            (.leftShoulder, .leftHip),
            (.rightShoulder, .rightHip),
            (.leftHip, .rightHip),
            // Legs
            (.leftHip, .leftKnee),
            (.leftKnee, .leftAnkle),
            (.rightHip, .rightKnee),
            (.rightKnee, .rightAnkle),
            // Head connections
            (.nose, .leftShoulder),
            (.nose, .rightShoulder)
        ]
        ForEach(jointPairs.indices, id: \ .self) { i in
            let (joint1, joint2) = jointPairs[i]
            if let p1 = points[joint1], let p2 = points[joint2] {
                Path { path in
                    let x1 = useFrontCamera ? (1.0 - p1.x) * geometry.size.width : p1.x * geometry.size.width
                    let y1 = (1.0 - p1.y) * geometry.size.height
                    let x2 = useFrontCamera ? (1.0 - p2.x) * geometry.size.width : p2.x * geometry.size.width
                    let y2 = (1.0 - p2.y) * geometry.size.height
                    path.move(to: CGPoint(x: x1, y: y1))
                    path.addLine(to: CGPoint(x: x2, y: y2))
                }
                .stroke(Color.orange, lineWidth: 3)
            }
        }
        // Draw pose points with labels
        ForEach(Array(points.keys), id: \.rawValue) { joint in
            if let point = points[joint] {
                ZStack {
                    Circle()
                        .fill(getJointColor(confidence: point.confidence))
                        .frame(width: max(8, CGFloat(point.confidence) * 16), height: max(8, CGFloat(point.confidence) * 16))
                    
                    // Add joint labels for key points
                    if shouldShowLabel(for: joint) {
                        Text(getJointLabel(joint))
                            .font(.caption2)
                            .foregroundColor(.white)
                            .padding(2)
                            .background(Color.black.opacity(0.7))
                            .cornerRadius(4)
                            .offset(x: 0, y: -20)
                    }
                }
                .position(
                    x: useFrontCamera ? (1.0 - point.x) * geometry.size.width : point.x * geometry.size.width,
                    y: (1.0 - point.y) * geometry.size.height
                )
            }
        }
    }
    
    @ViewBuilder
    private func angleView(in geometry: GeometryProxy) -> some View {
        if let angle = currentAngle {
            Text(String(format: "%.1f°", angle))
                .font(.caption)
                .foregroundColor(.white)
                .padding(4)
                .background(Color.black.opacity(0.7))
                .cornerRadius(4)
                .position(
                    x: geometry.size.width - 50,
                    y: 50
                )
        }
    }
    
    @ViewBuilder
    private func shotIndicator(in geometry: GeometryProxy) -> some View {
        if shotDetected {
            Text("Shot Detected!")
                .font(.headline)
                .foregroundColor(.green)
                .padding(8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(8)
                .position(
                    x: geometry.size.width / 2,
                    y: 50
                )
        }
    }
    
    @ViewBuilder
    private func userDefinedHoopAreaView(in geometry: GeometryProxy) -> some View {
        if let userDefinedHoopArea = userDefinedHoopArea {
            Rectangle()
                .stroke(Color.yellow, lineWidth: 2)
                .frame(
                    width: userDefinedHoopArea.width * geometry.size.width,
                    height: userDefinedHoopArea.height * geometry.size.height
                )
                .position(
                    x: (userDefinedHoopArea.minX + userDefinedHoopArea.width/2) * geometry.size.width,
                    y: (userDefinedHoopArea.minY + userDefinedHoopArea.height/2) * geometry.size.height
                )
        }
    }
    
    @ViewBuilder
    private func ballBoxView(in geometry: GeometryProxy) -> some View {
        if let ballBox = ballBox {
            Rectangle()
                .stroke(Color.blue, lineWidth: 2)
                .frame(
                    width: ballBox.width * geometry.size.width,
                    height: ballBox.height * geometry.size.height
                )
                .position(
                    x: (ballBox.minX + ballBox.width/2) * geometry.size.width,
                    y: (ballBox.minY + ballBox.height/2) * geometry.size.height
                )
        }
    }
    
    @ViewBuilder
    private func ballTrajectoryView(in geometry: GeometryProxy) -> some View {
        if !ballTrajectory.isEmpty {
            Path { path in
                path.move(to: CGPoint(x: ballTrajectory[0].x * geometry.size.width, y: ballTrajectory[0].y * geometry.size.height))
                for point in ballTrajectory {
                    path.addLine(to: CGPoint(x: point.x * geometry.size.width, y: point.y * geometry.size.height))
                }
            }
            .stroke(Color.cyan, lineWidth: 2)
        }
    }
    
    // Helper function for confidence-based coloring
    private func getJointColor(confidence: Float) -> Color {
        if confidence > 0.8 {
            return .green        // High confidence - bright green
        } else if confidence > 0.6 {
            return .yellow       // Medium confidence - yellow
        } else if confidence > 0.4 {
            return .orange       // Low confidence - orange
        } else {
            return .red          // Very low confidence - red
        }
    }
    
    // Helper functions for joint labels
    private func shouldShowLabel(for joint: VNHumanBodyPoseObservation.JointName) -> Bool {
        let labelJoints: [VNHumanBodyPoseObservation.JointName] = [
            .nose, .leftWrist, .rightWrist, .leftAnkle, .rightAnkle,
            .leftShoulder, .rightShoulder, .leftElbow, .rightElbow
        ]
        return labelJoints.contains(joint)
    }
    
    private func getJointLabel(_ joint: VNHumanBodyPoseObservation.JointName) -> String {
        switch joint {
        case .nose: return "Head"
        case .leftWrist: return "L Hand"
        case .rightWrist: return "R Hand"
        case .leftAnkle: return "L Foot"
        case .rightAnkle: return "R Foot"
        case .leftShoulder: return "L Shoulder"
        case .rightShoulder: return "R Shoulder"
        case .leftElbow: return "L Elbow"
        case .rightElbow: return "R Elbow"
        default: return ""
        }
    }
    
    // Calculate and display joint angles
    private func calculateAngles(in geometry: GeometryProxy) -> some View {
        VStack {
            // Right arm angle (shoulder-elbow-hand)
            if let shoulder = getJoint(.rightShoulder),
               let elbow = getJoint(.rightElbow),
               let hand = getJoint(.rightWrist) {
                let angle = calculateAngle(p1: shoulder, p2: elbow, p3: hand)
                Text("R Arm: \(String(format: "%.0f°", angle))")
                    .font(.caption)
                    .foregroundColor(.yellow)
                    .padding(4)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(4)
            }
            
            // Left arm angle (shoulder-elbow-hand)
            if let shoulder = getJoint(.leftShoulder),
               let elbow = getJoint(.leftElbow),
               let hand = getJoint(.leftWrist) {
                let angle = calculateAngle(p1: shoulder, p2: elbow, p3: hand)
                Text("L Arm: \(String(format: "%.0f°", angle))")
                    .font(.caption)
                    .foregroundColor(.yellow)
                    .padding(4)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(4)
            }
        }
        .position(x: geometry.size.width - 80, y: 100)
    }
    
    private func getJoint(_ jointName: VNHumanBodyPoseObservation.JointName) -> CGPoint? {
        guard let point = points[jointName] else { return nil }
        return CGPoint(x: point.x, y: point.y)
    }
    
    private func calculateAngle(p1: CGPoint, p2: CGPoint, p3: CGPoint) -> CGFloat {
        let v1 = CGVector(dx: p1.x - p2.x, dy: p1.y - p2.y)
        let v2 = CGVector(dx: p3.x - p2.x, dy: p3.y - p2.y)
        
        let dot = v1.dx * v2.dx + v1.dy * v2.dy
        let mag1 = sqrt(v1.dx * v1.dx + v1.dy * v1.dy)
        let mag2 = sqrt(v2.dx * v2.dx + v2.dy * v2.dy)
        
        guard mag1 > 0 && mag2 > 0 else { return 0 }
        
        let cosAngle = dot / (mag1 * mag2)
        let clampedCos = max(-1.0, min(1.0, cosAngle))
        let angleRadians = acos(clampedCos)
        
        return angleRadians * 180.0 / .pi
    }
    
    // Debug information overlay
    private func debugInfo(in geometry: GeometryProxy) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Joints: \(points.count)")
                .font(.caption2)
                .foregroundColor(.white)
            
            Text("Basketballs: \(hoopBoundingBoxes.filter { $0.label == "basketball" }.count)")
                .font(.caption2)
                .foregroundColor(.white)
            
            Text("Rims: \(hoopBoundingBoxes.filter { $0.label.lowercased().contains("rim") || $0.label.lowercased().contains("hoop") }.count)")
                .font(.caption2)
                .foregroundColor(.white)
            
            if let person = personLocation {
                Text("Person: ✓")
                    .font(.caption2)
                    .foregroundColor(.green)
            } else {
                Text("Person: ✗")
                    .font(.caption2)
                    .foregroundColor(.red)
            }
            
            if shotDetected {
                Text("SHOT!")
                    .font(.caption2)
                    .foregroundColor(.yellow)
            }
        }
        .padding(8)
        .background(Color.black.opacity(0.7))
        .cornerRadius(8)
        .position(x: 80, y: 60)
    }
    
    // Show 640x640 segment grid for debugging
    private func segmentGrid(in geometry: GeometryProxy) -> some View {
        let segmentSize: CGFloat = 640
        let screenWidth = geometry.size.width
        let screenHeight = geometry.size.height
        
        // Calculate segment dimensions
        let segmentWidth = min(segmentSize, screenWidth)
        let segmentHeight = min(segmentSize, screenHeight)
        
        // Calculate number of segments
        let cols = Int(ceil(screenWidth / segmentWidth))
        let rows = Int(ceil(screenHeight / segmentHeight))
        
        return ZStack {
            verticalLines(screenWidth: screenWidth, screenHeight: screenHeight, segmentWidth: segmentWidth, cols: cols)
            horizontalLines(screenWidth: screenWidth, screenHeight: screenHeight, segmentHeight: segmentHeight, rows: rows)
            segmentLabels(segmentWidth: segmentWidth, segmentHeight: segmentHeight, rows: rows, cols: cols)
        }
    }
    
    private func verticalLines(screenWidth: CGFloat, screenHeight: CGFloat, segmentWidth: CGFloat, cols: Int) -> some View {
        ForEach(0..<(cols + 1), id: \.self) { i in
            let x = CGFloat(i) * segmentWidth
            Rectangle()
                .fill(Color.red.opacity(0.3))
                .frame(width: 1, height: screenHeight)
                .position(x: x, y: screenHeight / 2)
        }
    }
    
    private func horizontalLines(screenWidth: CGFloat, screenHeight: CGFloat, segmentHeight: CGFloat, rows: Int) -> some View {
        ForEach(0..<(rows + 1), id: \.self) { i in
            let y = CGFloat(i) * segmentHeight
            Rectangle()
                .fill(Color.red.opacity(0.3))
                .frame(width: screenWidth, height: 1)
                .position(x: screenWidth / 2, y: y)
        }
    }
    
    private func segmentLabels(segmentWidth: CGFloat, segmentHeight: CGFloat, rows: Int, cols: Int) -> some View {
        ForEach(0..<rows, id: \.self) { row in
            ForEach(0..<cols, id: \.self) { col in
                Text("\(row),\(col)")
                    .font(.caption2)
                    .foregroundColor(.red)
                    .padding(2)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(4)
                    .position(
                        x: CGFloat(col) * segmentWidth + 20,
                        y: CGFloat(row) * segmentHeight + 20
                    )
            }
        }
    }
    
    // Display current shot phase prominently
    @ViewBuilder
    private func shotPhaseDisplay(in geometry: GeometryProxy) -> some View {
        VStack(spacing: 8) {
            // Shot Phase Display
            HStack {
                Text("📊 PHASE:")
                    .font(.caption.bold())
                    .foregroundColor(.white)
                
                Text(currentShotPhase)
                    .font(.title2.bold())
                    .foregroundColor(getPhaseColor(currentShotPhase))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 4)
                    .background(getPhaseColor(currentShotPhase).opacity(0.2))
                    .cornerRadius(8)
            }
            
            // Shooting Hand Display
            if shootingHand != "UNKNOWN" {
                HStack {
                    Text("✋ HAND:")
                        .font(.caption.bold())
                        .foregroundColor(.white)
                    
                    Text(shootingHand)
                        .font(.headline.bold())
                        .foregroundColor(shootingHand == "LEFT" ? .purple : .cyan)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background((shootingHand == "LEFT" ? Color.purple : Color.cyan).opacity(0.2))
                        .cornerRadius(6)
                }
            }
        }
        .padding(12)
        .background(Color.black.opacity(0.8))
        .cornerRadius(12)
        .position(x: geometry.size.width / 2, y: 80)
    }
    
    // Get color for different shot phases
    private func getPhaseColor(_ phase: String) -> Color {
        switch phase {
        case "Ready":
            return .gray
        case "Loading", "Preparation":
            return .yellow
        case "Release":
            return .red
        case "Follow", "followThrough":
            return .orange
        case "Complete":
            return .green
        default:
            return .white
        }
    }
} 