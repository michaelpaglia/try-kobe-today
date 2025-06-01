//
//  ContentView.swift
//  kobe-app
//
//  Created by Michael Paglia on 5/31/25.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Image(systemName: "globe")
                    .imageScale(.large)
                    .foregroundStyle(.tint)
                Text("Hello, world!")
                NavigationLink(destination: BasketballTrainerView()) {
                    Text("Start Basketball Trainer")
                        .font(.headline)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
            }
            .padding()
            .navigationTitle("Kobe App")
        }
    }
}

#Preview {
    ContentView()
}
