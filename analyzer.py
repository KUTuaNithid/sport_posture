import cv2
import math
import numpy as np
import streamlit as st
from ultralytics import YOLO
from moviepy import VideoFileClip
import os
@st.cache_resource
def load_pose_model(model_name='yolov8n-pose.pt', use_intel_gpu=False):
    # Load YOLO Pose model
    model = YOLO(model_name)
    if use_intel_gpu:
        import os
        ov_dir = model_name.replace('.pt', '_openvino_model')
        if not os.path.exists(ov_dir):
            st.info(f"Exporting {model_name} to OpenVINO for Intel GPU... This happens once.")
            model.export(format='openvino')
        return YOLO(ov_dir, task='pose')
    return model

def calculate_angle(A, B, C):
    """
    Calculate angle at vertex B formed by points A, B, and C.
    Points are (x, y). Returns angle in degrees.
    Formula: arccos( (BA dot BC) / (|BA| * |BC|) )
    """
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])
    
    norm_BA = np.linalg.norm(BA)
    norm_BC = np.linalg.norm(BC)
    
    if norm_BA == 0 or norm_BC == 0:
        return 0.0
        
    cos_theta = np.dot(BA, BC) / (norm_BA * norm_BC)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)

@st.cache_data(show_spinner=False)
def analyze_video(video_path, model_name='yolov8n-pose.pt', use_intel_gpu=False, show_keypoint_numbers=False, show_keypoint_confidence=False, min_confidence=0.5):
    model = load_pose_model(model_name, use_intel_gpu)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_output_path = "temp_output.mp4"
    final_output_path = "final_output.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    
    frames_data = [] 
    
    # Pass 1: Extract Keypoints for all frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Using Ultralytics YOLO inference
        if use_intel_gpu:
            results = model(frame, device='cpu', verbose=False)
        else:
            results = model(frame, verbose=False)
        keypoints = None
        if len(results) > 0 and len(results[0].keypoints) > 0:
            kpts_xy = results[0].keypoints.xy.cpu().numpy()[0]
            kpts_conf = results[0].keypoints.conf.cpu().numpy()[0]
            
            # Filter by confidence and attach it to keypoint data
            keypoints = []
            for (x, y), conf in zip(kpts_xy, kpts_conf):
                if conf >= min_confidence:
                    keypoints.append([x, y, conf])
                else:
                    keypoints.append([0.0, 0.0, 0.0]) # 0,0 is ignored by our logic
            keypoints = np.array(keypoints)
        
        frames_data.append({
            'frame': frame,
            'keypoints': keypoints
        })
        
    cap.release()
    
    # Pass 2: Analyze kinematics
    # COCO Keypoint IDs
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16
    
    # Connections for drawing the skeleton
    SKELETON_CONNECTIONS = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

    foot_strike_idx = 0
    lowest_y = -1
    
    # Find frame with lowest ankle position (highest Y pixel value)
    for idx, data in enumerate(frames_data):
        kpts = data['keypoints']
        if kpts is not None and len(kpts) >= 17:
            l_ankle_y = kpts[L_ANKLE][1]
            r_ankle_y = kpts[R_ANKLE][1]
            y_max = max(l_ankle_y, r_ankle_y)
            if y_max > lowest_y:
                lowest_y = y_max
                foot_strike_idx = idx

    # Refine foot strike using horizontal velocity drop
    search_window = min(len(frames_data)-1, foot_strike_idx + int(fps * 0.2))
    refined_strike_idx = foot_strike_idx
    min_x_vel = float('inf')
    
    for i in range(max(1, foot_strike_idx - 5), search_window):
        kpts_prev = frames_data[i-1]['keypoints']
        kpts_curr = frames_data[i]['keypoints']
        if kpts_prev is not None and kpts_curr is not None:
            # Assuming lowest leg is the leading leg
            ankle_idx = L_ANKLE if kpts_curr[L_ANKLE][1] > kpts_curr[R_ANKLE][1] else R_ANKLE
            x_vel = abs(kpts_curr[ankle_idx][0] - kpts_prev[ankle_idx][0])
            
            if x_vel < min_x_vel:
                min_x_vel = x_vel
                refined_strike_idx = i
                
    foot_strike_frame_idx = refined_strike_idx
    
    # --- Lock in the active leg side using the foot strike frame ---
    strike_kpts = frames_data[foot_strike_frame_idx]['keypoints']
    if strike_kpts is not None and len(strike_kpts) >= 17:
        locked_idx_hip = L_HIP if strike_kpts[L_ANKLE][1] > strike_kpts[R_ANKLE][1] else R_HIP
        locked_idx_knee = L_KNEE if locked_idx_hip == L_HIP else R_KNEE
        locked_idx_ankle = L_ANKLE if locked_idx_hip == L_HIP else R_ANKLE
        locked_idx_shoulder = L_SHOULDER if locked_idx_hip == L_HIP else R_SHOULDER
    else:
        locked_idx_hip, locked_idx_knee, locked_idx_ankle, locked_idx_shoulder = L_HIP, L_KNEE, L_ANKLE, L_SHOULDER
        
    # Determine running direction across the video
    start_hip_x, end_hip_x = -1, -1
    for data in frames_data:
        if data['keypoints'] is not None and len(data['keypoints']) >= 17:
            if start_hip_x == -1:
                start_hip_x = data['keypoints'][locked_idx_hip][0]
            end_hip_x = data['keypoints'][locked_idx_hip][0]
            
    running_left_to_right = end_hip_x > start_hip_x
        
    # Calculate metrics for ALL valid frames first
    for data in frames_data:
        kpts = data['keypoints']
        data['torso_lean'] = 0.0
        data['knee_flexion'] = 0.0
        data['overstride_distance'] = 0.0
        data['overstride_ratio'] = 0.0
        data['valid_metrics'] = False
        
        if kpts is not None and len(kpts) >= 17:
            # Assume active side locked from foot_strike frame
            hip = kpts[locked_idx_hip]
            knee = kpts[locked_idx_knee]
            ankle = kpts[locked_idx_ankle]
            shoulder = kpts[locked_idx_shoulder]
            
            # Ensure high confidence (non-zeroed out)
            if hip[0] > 0 and knee[0] > 0 and ankle[0] > 0 and shoulder[0] > 0:
                data['valid_metrics'] = True
                
                # Torso Lean 
                vertical_point = [hip[0], hip[1] - 100]
                lean_angle = calculate_angle(shoulder, hip, vertical_point)
                
                # Assign sign based on running direction
                is_leaning_forward = (shoulder[0] > hip[0]) if running_left_to_right else (shoulder[0] < hip[0])
                data['torso_lean'] = lean_angle if is_leaning_forward else -lean_angle
                
                # Knee Flexion (External angle)
                data['knee_flexion'] = 180.0 - calculate_angle(hip, knee, ankle)
                
                # Overstriding: Horizontal distance. 
                data['overstride_distance'] = abs(ankle[0] - hip[0])
                
                # Normalize overstride by leg length
                leg_length = np.linalg.norm(np.array([hip[0] - knee[0], hip[1] - knee[1]])) + \
                             np.linalg.norm(np.array([knee[0] - ankle[0], knee[1] - ankle[1]]))
                if leg_length > 0:
                    data['overstride_ratio'] = data['overstride_distance'] / leg_length

    # Calculate metrics by averaging a window around foot strike
    window_start = max(0, foot_strike_frame_idx - 2)
    window_end = min(len(frames_data), foot_strike_frame_idx + 3)
    
    torso_leans = []
    knee_flexions = []
    overstride_ratios = []
    
    for i in range(window_start, window_end):
        if i < len(frames_data) and frames_data[i]['valid_metrics']:
            torso_leans.append(frames_data[i]['torso_lean'])
            knee_flexions.append(frames_data[i]['knee_flexion'])
            overstride_ratios.append(frames_data[i]['overstride_ratio'])
            
    # Compute averages
    torso_lean = np.mean(torso_leans) if torso_leans else 0.0
    knee_flexion = np.mean(knee_flexions) if knee_flexions else 0.0
    overstride_ratio = np.mean(overstride_ratios) if overstride_ratios else 0.0
    
    is_overstriding = False
    
    if len(overstride_ratios) > 0:
        # threshold for POC: > 8% of leg length indicates overstride
        if overstride_ratio > 0.08: 
            is_overstriding = True
            
    debug_image_paths = []
    import time
    timestamp_base = int(time.time())
            
    # Draw overlays
    for i, data in enumerate(frames_data):
        frame = data['frame'].copy()
        kpts = data['keypoints']
        
        if kpts is not None:
            # Draw Skeleton lines
            for conn in SKELETON_CONNECTIONS:
                if len(kpts) > max(conn):
                    pt1 = (int(kpts[conn[0]][0]), int(kpts[conn[0]][1]))
                    pt2 = (int(kpts[conn[1]][0]), int(kpts[conn[1]][1]))
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
            
            # Draw Keypoints
            for pt_idx, pt in enumerate(kpts):
                x, y = int(pt[0]), int(pt[1])
                conf = float(pt[2]) if len(pt) > 2 else 0.0
                if x > 0 and y > 0:
                    text_offset = -10
                    if show_keypoint_numbers:
                        cv2.putText(frame, str(pt_idx), (x, y + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        text_offset -= 15
                    if show_keypoint_confidence:
                        cv2.putText(frame, f"{conf:.2f}", (x, y + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    if not show_keypoint_numbers:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
        # If this frame is part of the calculation window, draw the metrics measured on THIS frame
        if window_start <= i < window_end and data['valid_metrics']:
            if i == foot_strike_frame_idx:
                cv2.putText(frame, "FOOT STRIKE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                cv2.putText(frame, f"AVERAGING WINDOW: FRAME {i - window_start + 1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
            
            kpts = data['keypoints']
            # Locked variables
            hip = (int(kpts[locked_idx_hip][0]), int(kpts[locked_idx_hip][1]))
            knee = (int(kpts[locked_idx_knee][0]), int(kpts[locked_idx_knee][1]))
            ankle = (int(kpts[locked_idx_ankle][0]), int(kpts[locked_idx_ankle][1]))
            shoulder = (int(kpts[locked_idx_shoulder][0]), int(kpts[locked_idx_shoulder][1]))
            
            # 1. Torso Lean (Vertical Line + Torso Line)
            vertical_top = (hip[0], hip[1] - 150)
            cv2.line(frame, hip, vertical_top, (0, 255, 255), 2, cv2.LINE_AA) # Vertical ref
            cv2.line(frame, hip, shoulder, (255, 0, 0), 3, cv2.LINE_AA) # Torso line
            cv2.putText(frame, f"Torso Lean: {data['torso_lean']:.1f} deg", (hip[0] + 20, hip[1] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            # 2. Knee Flexion (Hip->Knee, Knee->Ankle)
            cv2.line(frame, hip, knee, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(frame, knee, ankle, (255, 0, 0), 3, cv2.LINE_AA)
            
            # Extend thigh line (Hip->Knee) for external angle visualization
            dx_thigh = knee[0] - hip[0]
            dy_thigh = knee[1] - hip[1]
            len_thigh = math.hypot(dx_thigh, dy_thigh)
            len_shin = math.hypot(ankle[0] - knee[0], ankle[1] - knee[1])
            
            if len_thigh > 0:
                ext_x = int(knee[0] + (dx_thigh / len_thigh) * len_shin)
                ext_y = int(knee[1] + (dy_thigh / len_thigh) * len_shin)
                cv2.line(frame, knee, (ext_x, ext_y), (0, 165, 255), 2, cv2.LINE_AA) # Orange extended line
                
            cv2.putText(frame, f"Knee: {data['knee_flexion']:.1f} deg", (knee[0] + 20, knee[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 3. Overstride (Horizontal line floor)
            floor_y = ankle[1]
            hip_proj = (hip[0], floor_y)
            cv2.line(frame, hip, hip_proj, (0, 255, 255), 2, cv2.LINE_AA) # vertical drop from hip
            
            is_frame_overstride = data.get('overstride_ratio', 0) > 0.08
            cv2.line(frame, hip_proj, ankle, (0, 0, 255) if is_frame_overstride else (0, 255, 0), 3, cv2.LINE_AA) # horizontal ground line
            
            cv2.putText(frame, f"Overstride: {data.get('overstride_ratio', 0)*100:.1f}% leg", (hip_proj[0], floor_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_frame_overstride else (0, 255, 0), 2)
            
            # Save debug image
            os.makedirs("debug_images", exist_ok=True)
            debug_image_path = f"debug_images/window_frame_{timestamp_base}_{i}.jpg"
            cv2.imwrite(debug_image_path, frame)
            debug_image_paths.append(debug_image_path)
            
        out.write(frame)
        
    out.release()
    
    # Convert debug video to H.264 using moviepy
    clip = VideoFileClip(temp_output_path)
    clip.write_videofile(final_output_path, codec="libx264", audio=False, logger=None)
    clip.close()
    
    # Cleanup intermediate
    try:
        os.remove(temp_output_path)
    except:
        pass
        

        
    return {
        'output_video_path': final_output_path,
        'debug_image_paths': debug_image_paths,
        'torso_lean': torso_lean,
        'knee_flexion': knee_flexion,
        'overstride_ratio': overstride_ratio,
        'is_overstriding': is_overstriding
    }
