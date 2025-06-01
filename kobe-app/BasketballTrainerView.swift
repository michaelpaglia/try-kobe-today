import SwiftUI

struct BasketballTrainerView: View {
    @State private var posePoints: [CGPoint] = []
    @State private var hoopLocation: CGPoint? = nil
    @State private var showHoopPrompt: Bool = true
    @State private var shotCount: Int = 0
    @State private var showShotMessage: Bool = false
    @State private var lastWristY: CGFloat? = nil
    @State private var lastElbowY: CGFloat? = nil
    @State private var lastShoulderY: CGFloat? = nil
    @State private var armExtended: Bool = false
    @State private var shotState: ShotState = .waitingForLoad
    @State private var useFrontCamera: Bool = false
    @State private var segmentationMask: CGImage? = nil
    
    // Indices for right arm keypoints in Vision's output (approximate order)
    let rightWristIndex = 14
    let rightElbowIndex = 13
    let rightShoulderIndex = 12
    
    enum ShotState {
        case waitingForLoad, loaded, extended
    }
    
    var body: some View {
        ZStack {
            CameraView(posePoints: $posePoints, useFrontCamera: useFrontCamera, segmentationMask: $segmentationMask)
                .edgesIgnoringSafeArea(.all)
            PoseOverlayView(points: posePoints)
                .allowsHitTesting(false)
            SegmentationOverlayView(mask: segmentationMask)
                .edgesIgnoringSafeArea(.all)
            if showHoopPrompt {
                Color.black.opacity(0.3)
                    .edgesIgnoringSafeArea(.all)
                    .onTapGesture { location in
                        hoopLocation = location
                        showHoopPrompt = false
                    }
                    .overlay(
                        VStack {
                            Text("Tap the screen where the hoop is")
                                .font(.title2)
                                .foregroundColor(.white)
                                .padding()
                                .background(Color.black.opacity(0.7))
                                .cornerRadius(10)
                        }
                    )
            }
            if let hoop = hoopLocation {
                GeometryReader { geo in
                    Circle()
                        .stroke(Color.red, lineWidth: 3)
                        .frame(width: 40, height: 40)
                        .position(hoop)
                }
                .allowsHitTesting(false)
            }
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
                Text("Basketball Trainer")
                    .font(.title)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                HStack {
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
            }
        }
        .onChange(of: posePoints) { newPoints in
            detectShot(from: newPoints)
        }
    }
    
    func detectShot(from points: [CGPoint]) {
        guard points.count > rightWristIndex,
              points.count > rightElbowIndex,
              points.count > rightShoulderIndex else { return }
        let wristY = points[rightWristIndex].y
        let elbowY = points[rightElbowIndex].y
        let shoulderY = points[rightShoulderIndex].y
        
        switch shotState {
        case .waitingForLoad:
            // Wait for wrist to be below shoulder (load phase)
            if wristY > shoulderY + 30 {
                shotState = .loaded
            }
        case .loaded:
            // Wait for wrist to move above shoulder (extend phase)
            if wristY < shoulderY - 20 {
                shotState = .extended
            }
        case .extended:
            // Wait for wrist to drop again (release/follow-through)
            if wristY > shoulderY + 10 {
                shotCount += 1
                showShotMessage = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    showShotMessage = false
                }
                shotState = .waitingForLoad
            }
        }
        lastWristY = wristY
        lastElbowY = elbowY
        lastShoulderY = shoulderY
    }
}

#Preview {
    BasketballTrainerView()
} 