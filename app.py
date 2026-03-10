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
    is_treadmill = st.checkbox("Running on a Treadmill? (Changes foot strike physics)")
    running_direction = st.radio("Runner Direction (Important for treadmill videos)", ["Left to Right", "Right to Left"])
    if st.button("Analyze Posture"):
        with st.spinner("Analyzing video... This might take a minute."):
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_input_path = tfile.name
            tfile.close()
            
            # Analyze
            result = analyze_video(temp_input_path, model_choice, use_intel_gpu, show_keypoint_numbers, show_keypoint_confidence, min_confidence, running_direction, is_treadmill)
            
            st.success("Analysis Complete!")
            
            # Display debug location info
            if result.get('debug_run_dir'):
                abs_path = os.path.abspath(result['debug_run_dir'])
                st.info(f"📁 **Detailed frame-by-frame debug images saved to:** `{abs_path}`\n\nInside you will find `left_tracking` and `right_tracking` folders containing analysis for every single frame.")
                            
            # --- Per-Step Metrics Data Preparation ---
            step_metrics = result.get('step_metrics', [])
            left_steps = [s for s in step_metrics if s['leg'] == 'Left']
            right_steps = [s for s in step_metrics if s['leg'] == 'Right']
            
            def safe_mean(metrics_list, key):
                return sum(s[key] for s in metrics_list) / len(metrics_list) if metrics_list else 0.0
                
            l_lean = safe_mean(left_steps, 'torso_lean')
            r_lean = safe_mean(right_steps, 'torso_lean')
            l_knee = safe_mean(left_steps, 'knee_flexion')
            r_knee = safe_mean(right_steps, 'knee_flexion')
            l_over = safe_mean(left_steps, 'overstride_ratio')
            r_over = safe_mean(right_steps, 'overstride_ratio')
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Debug Video")
                st.video(result['output_video_path'])
                
                if step_metrics:
                    st.subheader("Averages by Leg (Symmetry)")
                    leg_col1, leg_col2 = st.columns(2)
                    with leg_col1:
                        st.markdown("**Left Leg Averages**")
                        st.metric("Torso Lean", f"{l_lean:.1f}°")
                        st.metric("Knee Flexion", f"{l_knee:.1f}°")
                        st.metric("Overstride Ratio", f"{l_over*100:.1f}%")
                    with leg_col2:
                        st.markdown("**Right Leg Averages**")
                        st.metric("Torso Lean", f"{r_lean:.1f}°")
                        st.metric("Knee Flexion", f"{r_knee:.1f}°")
                        st.metric("Overstride Ratio", f"{r_over*100:.1f}%")
                        
                    st.subheader("Consistency Over Time (Per Step)")
                    import pandas as pd
                    df = pd.DataFrame(step_metrics)
                    df['Step Number'] = df['step_number']
                    df.set_index('Step Number', inplace=True)
                    
                    st.markdown("**Torso Lean consistency**")
                    st.line_chart(df[['torso_lean']])
                    
                    st.markdown("**Knee Flexion consistency**")
                    st.line_chart(df[['knee_flexion']])
                    
                    st.markdown("**Overstride Ratio % consistency**")
                    df['overstride_percentage'] = df['overstride_ratio'] * 100
                    st.line_chart(df[['overstride_percentage']])
            
            with col2:
                st.subheader("Overall Run Averages")
                st.metric("Torso Lean", f"{result['torso_lean']:.1f}°")
                st.metric("Knee Flexion", f"{result['knee_flexion']:.1f}°")
                st.metric("Overstride Ratio", f"{result['overstride_ratio']*100:.1f}% of Leg Length")
                
                st.subheader("Feedback")
                if result['is_overstriding']:
                    st.warning("⚠️ **Overstriding Detected!** On average, your foot is landing too far in front of your body's center of mass.")
                    st.write("💡 *Actionable Advice*: Try to increase your cadence (steps per minute). A higher cadence naturally shortens stride length and brings your foot strike safely back under your hips.")
                else:
                    st.success("✅ **Good Stride Length!** Your foot is landing nicely under your center of mass.")
            
            try:
                os.remove(temp_input_path)
            except Exception:
                pass
