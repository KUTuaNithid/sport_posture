import streamlit as st
import tempfile
import os
import math
import numpy as np
import PIL.Image
import plotly.express as px
from streamlit_drawable_canvas import st_canvas
from analyzer import analyze_video, load_pose_model, get_video_info, get_frame, get_frame_thumbnails, compute_single_frame_metrics, annotate_frame

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
    
    # Save to session_state so it persists across slider interactions
    if 'temp_video_path' not in st.session_state or st.session_state.get('last_file_name') != uploaded_file.name:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        st.session_state['temp_video_path'] = tfile.name
        st.session_state['last_file_name'] = uploaded_file.name
        
    video_path = st.session_state['temp_video_path']
    
    is_treadmill = st.checkbox("Running on a Treadmill? (Changes foot strike physics)")
    running_direction = st.radio("Runner Direction (Important for treadmill videos)", ["Left to Right", "Right to Left"])
    
    tab1, tab2 = st.tabs(["Full Video Analysis", "Single Frame Analyze Mode"])
    
    with tab1:
        if st.button("Start Full Analysis"):
            with st.spinner("Analyzing full video... This might take a minute."):
                # Analyze full video
                result = analyze_video(video_path, model_choice, use_intel_gpu, show_keypoint_numbers, show_keypoint_confidence, min_confidence, running_direction, is_treadmill)
            
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
                    
    with tab2:
        st.subheader("Interactive Frame Inspector")
        st.markdown("Seek to a specific frame using the 2-step visual picker, then draw lines to measure precise coordinates.")
        
        total_frames, fps, w, h = get_video_info(video_path)
        
        frame_idx = st.slider("Scrub Video to Select Frame", 0, max(0, total_frames - 1), 0)
        st.divider()
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            sf_side = st.radio("Target Leg", ["Left", "Right"], key="sf_side")
        with col_ctrl2:
            sf_show_overlay = st.checkbox("Show AI Metrics Overlay", value=True, key="sf_overlay")
        with col_ctrl3:
            sf_scale = st.number_input("Calibration (Line 1 Length in cm)", min_value=1.0, value=100.0, step=1.0, help="Draw a line over an object of known length to calibrate the drawing tool. Then input that length here.")
            
        frame_rgb = get_frame(video_path, frame_idx)
        if frame_rgb is not None:
            # Run inference on single frame immediately
            model = load_pose_model(model_choice, use_intel_gpu)
            metrics = compute_single_frame_metrics(frame_rgb, model, min_confidence, running_direction)
            
            if sf_show_overlay:
                display_img = annotate_frame(frame_rgb, metrics, sf_side, show_skeleton=True, show_metrics=True, show_keypoint_numbers=show_keypoint_numbers, show_keypoint_confidence=show_keypoint_confidence)
            else:
                display_img = frame_rgb
                
            # Scale for canvas
            canvas_w = 1000
            canvas_h = int(h * (canvas_w / w))
            pil_img = PIL.Image.fromarray(display_img)
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0)",
                stroke_width=3,
                stroke_color="#FF00FF", # Magenta to contrast with YOLO red/yellow
                background_image=pil_img,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="line",
                key=f"canvas_{frame_idx}",
            )
            
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                lines = [obj for obj in objects if obj["type"] == "line"]
                
                if len(lines) > 0:
                    st.divider()
                    st.subheader("Interactive Measurements")
                    calib_line = lines[0]
                    dx = calib_line["x2"] - calib_line["x1"]
                    dy = calib_line["y2"] - calib_line["y1"]
                    calib_pix_len = math.hypot(dx, dy)
                    
                    if calib_pix_len > 0:
                        cm_per_pixel = sf_scale / calib_pix_len
                        st.success(f"📏 **Calibration Active**: Line 1 was set to **{sf_scale} cm**. Output ratio: `{cm_per_pixel:.3f} cm/pixel`")
                        
                        m_cols = st.columns(len(lines))
                        for idx, line in enumerate(lines):
                            dx_l = line["x2"] - line["x1"]
                            dy_l = line["y2"] - line["y1"]
                            pix_len = math.hypot(dx_l, dy_l)
                            cm_len = pix_len * cm_per_pixel
                            angle = math.degrees(math.atan2(-dy_l, dx_l)) # inverse Y for standard angle
                            
                            with m_cols[idx % len(m_cols)]:
                                st.info(f"**Line {idx + 1}**\n\nDistance: **{cm_len:.1f} cm**\n\nAngle: **{angle:.1f}°**")
                                
                        if len(lines) >= 2:
                            dx1 = lines[0]["x2"] - lines[0]["x1"]
                            dy1 = lines[0]["y2"] - lines[0]["y1"]
                            dx2 = lines[1]["x2"] - lines[1]["x1"]
                            dy2 = lines[1]["y2"] - lines[1]["y1"]
                            
                            dot = dx1*dx2 + dy1*dy2
                            mag1 = math.hypot(dx1, dy1)
                            mag2 = math.hypot(dx2, dy2)
                            if mag1 * mag2 > 0:
                                cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
                                inter_angle = math.degrees(math.acos(cos_theta))
                                st.warning(f"📐 Absolute Angle between Line 1 and Line 2: **{inter_angle:.1f}°**")
        else:
            st.error(f"❌ Could not load Frame {frame_idx}! The video file might be corrupted or OpenCV failed to seek to this timestamp.")
