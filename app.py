import streamlit as st
import tempfile
import os
from analyzer import analyze_video

st.set_page_config(page_title="Posture Correction POC", layout="wide")

st.title("🏃 Running Posture Correction")
st.write("Upload a side-view video of your running to analyze your posture and get actionable feedback.")

uploaded_file = st.file_uploader("Upload Running Video (MP4/MOV)", type=["mp4", "mov", "avi"])
model_choice = st.selectbox("Select Pose Model", ["yolov8n-pose.pt", "yolo11n-pose.pt", "yolo26n-pose.pt", "yolo26x-pose.pt"])
use_intel_gpu = st.checkbox("Use Intel GPU Acceleration (OpenVINO)")
show_keypoint_numbers = st.checkbox("DEBUG: Show Keypoint Numbers", value=False)
show_keypoint_confidence = st.checkbox("DEBUG: Show Keypoint Confidence", value=False)
min_confidence = st.slider("Minimum Keypoint Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Filter out AI joint detections below this confidence level to reduce false positives.")

if uploaded_file is not None:
    st.video(uploaded_file)
    if st.button("Analyze Posture"):
        with st.spinner("Analyzing video... This might take a minute."):
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_input_path = tfile.name
            tfile.close()
            
            # Analyze
            result = analyze_video(temp_input_path, model_choice, use_intel_gpu, show_keypoint_numbers, show_keypoint_confidence, min_confidence)
            
            st.success("Analysis Complete!")
            
            # Display images sequentially 
            if result.get('debug_image_paths'):
                st.subheader("Frames Used for Averaging")
                cols = st.columns(len(result['debug_image_paths']))
                for idx, img_path in enumerate(result['debug_image_paths']):
                    with cols[idx]:
                        st.image(img_path, caption=f"Window Frame {idx+1}")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Debug Video")
                st.video(result['output_video_path'])
            
            with col2:
                st.subheader("Metrics at Foot Strike")
                st.metric("Torso Lean", f"{result['torso_lean']:.1f}°")
                st.metric("Knee Flexion", f"{result['knee_flexion']:.1f}°")
                st.metric("Overstride Ratio", f"{result['overstride_ratio']*100:.1f}% of Leg Length")
                
                st.subheader("Feedback")
                if result['is_overstriding']:
                    st.warning("⚠️ **Overstriding Detected!** Your foot is landing too far in front of your body's center of mass.")
                    st.write("💡 *Actionable Advice*: Try to increase your cadence (steps per minute). A higher cadence naturally shortens stride length and brings your foot strike safely back under your hips.")
                else:
                    st.success("✅ **Good Stride Length!** Your foot is landing nicely under your center of mass.")
            
            try:
                os.remove(temp_input_path)
            except Exception:
                pass
