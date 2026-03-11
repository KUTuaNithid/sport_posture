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
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames, fps, width, height

def get_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def get_frame_thumbnails(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    thumbnails = []
    # Sort indices to only read video forward
    sorted_indices = sorted(frame_indices)
    idx_map = {}
    
    current_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened() and current_idx <= sorted_indices[-1]:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in sorted_indices:
            # Keep resolution high enough for zooming
            h, w = frame.shape[:2]
            thumb_h = min(h, 720)
            thumb_w = int(w * (thumb_h / h))
            if thumb_h < h:
                thumb = cv2.resize(frame, (thumb_w, thumb_h))
            else:
                thumb = frame
            idx_map[current_idx] = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        current_idx += 1
    cap.release()
    
    # Return in original requested order
    for idx in frame_indices:
        thumbnails.append(idx_map.get(idx, None))
    return thumbnails

# COCO Keypoint IDs
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
SKELETON_CONNECTIONS = [(0,1), (0,2), (1,3), (2,4), (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]

def compute_single_frame_metrics(frame, model, min_confidence=0.5, running_direction="Left to Right"):
    # Inference
    results = model(frame, verbose=False)
    keypoints = None
    if len(results) > 0 and len(results[0].keypoints) > 0:
        kpts_xy = results[0].keypoints.xy.cpu().numpy()[0]
        kpts_conf = results[0].keypoints.conf.cpu().numpy()[0]
        
        keypoints = []
        for (x, y), conf in zip(kpts_xy, kpts_conf):
            if conf >= min_confidence:
                keypoints.append([x, y, conf])
            else:
                keypoints.append([0.0, 0.0, 0.0])
        keypoints = np.array(keypoints)
        
    metrics = {'Left': {}, 'Right': {}, 'keypoints': keypoints}
    running_left_to_right = (running_direction == "Left to Right")
    
    if keypoints is not None and len(keypoints) >= 17:
        for side, idxs in [('Left', (L_HIP, L_KNEE, L_ANKLE, L_SHOULDER)), 
                           ('Right', (R_HIP, R_KNEE, R_ANKLE, R_SHOULDER))]:
            hip = keypoints[idxs[0]]
            knee = keypoints[idxs[1]]
            ankle = keypoints[idxs[2]]
            shoulder = keypoints[idxs[3]]
            
            if hip[0] > 0 and knee[0] > 0 and ankle[0] > 0 and shoulder[0] > 0:
                l_shoulder, r_shoulder = keypoints[L_SHOULDER], keypoints[R_SHOULDER]
                l_hip, r_hip = keypoints[L_HIP], keypoints[R_HIP]
                
                if l_shoulder[0] > 0 and r_shoulder[0] > 0 and l_hip[0] > 0 and r_hip[0] > 0:
                    virtual_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0]
                    virtual_hip = [(l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0]
                else:
                    virtual_shoulder = shoulder
                    virtual_hip = hip
                
                vertical_point = [virtual_hip[0], virtual_hip[1] - 100]
                lean_angle = calculate_angle(virtual_shoulder, virtual_hip, vertical_point)
                is_leaning_forward = (virtual_shoulder[0] > virtual_hip[0]) if running_left_to_right else (virtual_shoulder[0] < virtual_hip[0])
                torso_lean = lean_angle if is_leaning_forward else -lean_angle
                
                knee_flexion = 180.0 - calculate_angle(hip, knee, ankle)
                overstride_distance = abs(ankle[0] - hip[0])
                
                leg_length = np.linalg.norm(np.array([hip[0] - knee[0], hip[1] - knee[1]])) + \
                             np.linalg.norm(np.array([knee[0] - ankle[0], knee[1] - ankle[1]]))
                overstride_ratio = overstride_distance / leg_length if leg_length > 0 else 0
                
                metrics[side] = {
                    'valid': True, 'torso_lean': torso_lean, 'knee_flexion': knee_flexion,
                    'overstride_distance': overstride_distance, 'overstride_ratio': overstride_ratio,
                    'hip': hip, 'knee': knee, 'ankle': ankle, 'shoulder': shoulder,
                    'virtual_hip': virtual_hip, 'virtual_shoulder': virtual_shoulder
                }
    return metrics

def annotate_frame(frame_rgb, metrics, side='Left', show_skeleton=True, show_metrics=True, show_keypoint_numbers=False, show_keypoint_confidence=False):
    annotated = frame_rgb.copy()
    kpts = metrics.get('keypoints')
    
    if show_skeleton and kpts is not None:
        for conn in SKELETON_CONNECTIONS:
            if len(kpts) > max(conn):
                pt1 = (int(kpts[conn[0]][0]), int(kpts[conn[0]][1]))
                pt2 = (int(kpts[conn[1]][0]), int(kpts[conn[1]][1]))
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(annotated, pt1, pt2, (255, 255, 0), 2)
        
        for pt_idx, pt in enumerate(kpts):
            x, y = int(pt[0]), int(pt[1])
            conf = float(pt[2]) if len(pt) > 2 else 0.0
            if x > 0 and y > 0:
                text_offset = -10
                if show_keypoint_numbers:
                    cv2.putText(annotated, str(pt_idx), (x, y + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    text_offset -= 15
                if show_keypoint_confidence:
                    cv2.putText(annotated, f"{conf:.2f}", (x, y + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                if not show_keypoint_numbers:
                    cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)

    if show_metrics:
        m = metrics.get(side, {})
        if m.get('valid', False):
            hip_pt = (int(m['hip'][0]), int(m['hip'][1]))
            knee_pt = (int(m['knee'][0]), int(m['knee'][1]))
            ankle_pt = (int(m['ankle'][0]), int(m['ankle'][1]))
            v_hip = (int(m['virtual_hip'][0]), int(m['virtual_hip'][1]))
            v_shoulder = (int(m['virtual_shoulder'][0]), int(m['virtual_shoulder'][1]))
            
            # Torso Lean
            vertical_top = (v_hip[0], v_hip[1] - 150)
            cv2.line(annotated, v_hip, vertical_top, (0, 255, 255), 2, cv2.LINE_AA) 
            cv2.line(annotated, v_hip, v_shoulder, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, f"Lean: {m['torso_lean']:.1f} deg", (v_hip[0] + 20, v_hip[1] - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            # Knee Flexion
            cv2.line(annotated, hip_pt, knee_pt, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(annotated, knee_pt, ankle_pt, (255, 0, 0), 3, cv2.LINE_AA)
            dx_thigh = knee_pt[0] - hip_pt[0]
            dy_thigh = knee_pt[1] - hip_pt[1]
            len_thigh = math.hypot(dx_thigh, dy_thigh)
            len_shin = math.hypot(ankle_pt[0] - knee_pt[0], ankle_pt[1] - knee_pt[1])
            if len_thigh > 0:
                ext_x = int(knee_pt[0] + (dx_thigh / len_thigh) * len_shin)
                ext_y = int(knee_pt[1] + (dy_thigh / len_thigh) * len_shin)
                cv2.line(annotated, knee_pt, (ext_x, ext_y), (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Knee: {m['knee_flexion']:.1f} deg", (knee_pt[0] + 20, knee_pt[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
            # Overstride
            floor_y = ankle_pt[1]
            hip_proj = (hip_pt[0], floor_y)
            cv2.line(annotated, hip_pt, hip_proj, (0, 255, 255), 2, cv2.LINE_AA)
            is_frame_overstride = m['overstride_ratio'] > 0.08
            cv2.line(annotated, hip_proj, ankle_pt, (255, 0, 0) if is_frame_overstride else (0, 255, 0), 3, cv2.LINE_AA) # Red is BGR so RGB would be (255,0,0) wait cv2 functions use the array format. Actually, in Streamlit we are manipulating an RGB array. So (255, 0, 0) is Red, (0, 255, 0) is Green, (0, 0, 255) is Blue.
            color_overstride = (255, 0, 0) if is_frame_overstride else (0, 255, 0)
            cv2.line(annotated, hip_proj, ankle_pt, color_overstride, 3, cv2.LINE_AA)
            cv2.putText(annotated, f"Overstride: {m['overstride_ratio']*100:.1f}% leg", (hip_proj[0], floor_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_overstride, 2)
                        
    return annotated

@st.cache_data(show_spinner=False)
def analyze_video(video_path, model_name='yolov8n-pose.pt', use_intel_gpu=False, show_keypoint_numbers=False, show_keypoint_confidence=False, min_confidence=0.5, running_direction="Left to Right", is_treadmill=False):
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

    # 1. Detect Peaks using scipy
    import scipy.signal
    
    # Extract ankle Y arrays
    l_ankle_y = []
    r_ankle_y = []
    for data in frames_data:
        kpts = data['keypoints']
        if kpts is not None and len(kpts) >= 17:
            l_ankle_y.append(kpts[L_ANKLE][1])
            r_ankle_y.append(kpts[R_ANKLE][1])
        else:
            # use previous if possible, else 0
            l_ankle_y.append(l_ankle_y[-1] if len(l_ankle_y) > 0 else 0)
            r_ankle_y.append(r_ankle_y[-1] if len(r_ankle_y) > 0 else 0)
            
    l_ankle_y = np.array(l_ankle_y)
    r_ankle_y = np.array(r_ankle_y)
    
    # Distance of ~ 0.4 seconds between steps of the SAME leg
    min_dist = max(1, int(fps * 0.4)) 
    prominence_val = max(5, int(height * 0.02))
    
    l_peaks, _ = scipy.signal.find_peaks(l_ankle_y, distance=min_dist, prominence=prominence_val)
    r_peaks, _ = scipy.signal.find_peaks(r_ankle_y, distance=min_dist, prominence=prominence_val)
    
    # Refine function using horizontal velocity
    def refine_strike(peak_idx, leg):
        if is_treadmill:
            # On a treadmill, the lowest vertical position (the peak_idx itself) 
            # is the safest bet for mid-stance/strike.
            return peak_idx
            
        search_window = min(len(frames_data)-1, peak_idx + int(fps * 0.2))
        refined_idx = peak_idx
        min_x_vel = float('inf')
        ankle_idx = L_ANKLE if leg == 'Left' else R_ANKLE
        for i in range(max(1, peak_idx - 5), search_window):
            kpts_prev = frames_data[i-1]['keypoints']
            kpts_curr = frames_data[i]['keypoints']
            if kpts_prev is not None and kpts_curr is not None:
                x_vel = abs(kpts_curr[ankle_idx][0] - kpts_prev[ankle_idx][0])
                if x_vel < min_x_vel:
                    min_x_vel = x_vel
                    refined_idx = i
        return refined_idx

    step_events = []
    for p in l_peaks:
        step_events.append({'frame_idx': refine_strike(p, 'Left'), 'leg': 'Left'})
    for p in r_peaks:
        step_events.append({'frame_idx': refine_strike(p, 'Right'), 'leg': 'Right'})
        
    step_events.sort(key=lambda x: x['frame_idx'])
    
    # Determine running direction across the video from explicit UI selection (Handles treadmill edge case)
    locked_idx_hip = L_HIP
    running_left_to_right = (running_direction == "Left to Right")
        
    # Calculate metrics for ALL valid frames first for BOTH legs
    for data in frames_data:
        kpts = data['keypoints']
        data['metrics'] = {'Left': {}, 'Right': {}}
        
        if kpts is not None and len(kpts) >= 17:
            for side, idxs in [('Left', (L_HIP, L_KNEE, L_ANKLE, L_SHOULDER)), 
                               ('Right', (R_HIP, R_KNEE, R_ANKLE, R_SHOULDER))]:
                hip = kpts[idxs[0]]
                knee = kpts[idxs[1]]
                ankle = kpts[idxs[2]]
                shoulder = kpts[idxs[3]]
                
                # Ensure high confidence
                if hip[0] > 0 and knee[0] > 0 and ankle[0] > 0 and shoulder[0] > 0:
                    
                    # Virtual Spine calculations (Midpoints to cancel out arm swing/torso rotation)
                    l_shoulder = kpts[L_SHOULDER]
                    r_shoulder = kpts[R_SHOULDER]
                    l_hip = kpts[L_HIP]
                    r_hip = kpts[R_HIP]
                    
                    if l_shoulder[0] > 0 and r_shoulder[0] > 0 and l_hip[0] > 0 and r_hip[0] > 0:
                        virtual_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0]
                        virtual_hip = [(l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0]
                    else:
                        # Fallback to single side if opposite side is occluded
                        virtual_shoulder = shoulder
                        virtual_hip = hip
                    
                    # Torso Lean (using Virtual Spine)
                    vertical_point = [virtual_hip[0], virtual_hip[1] - 100]
                    lean_angle = calculate_angle(virtual_shoulder, virtual_hip, vertical_point)
                    
                    is_leaning_forward = (virtual_shoulder[0] > virtual_hip[0]) if running_left_to_right else (virtual_shoulder[0] < virtual_hip[0])
                    torso_lean = lean_angle if is_leaning_forward else -lean_angle
                    
                    # Knee Flexion (External angle)
                    knee_flexion = 180.0 - calculate_angle(hip, knee, ankle)
                    
                    # Overstriding
                    overstride_distance = abs(ankle[0] - hip[0])
                    leg_length = np.linalg.norm(np.array([hip[0] - knee[0], hip[1] - knee[1]])) + \
                                 np.linalg.norm(np.array([knee[0] - ankle[0], knee[1] - ankle[1]]))
                    overstride_ratio = overstride_distance / leg_length if leg_length > 0 else 0
                    
                    data['metrics'][side] = {
                        'valid': True,
                        'torso_lean': torso_lean,
                        'knee_flexion': knee_flexion,
                        'overstride_distance': overstride_distance,
                        'overstride_ratio': overstride_ratio,
                        'hip': hip, 'knee': knee, 'ankle': ankle, 'shoulder': shoulder,
                        'virtual_hip': virtual_hip, 'virtual_shoulder': virtual_shoulder
                    }

    # Calculate metrics for each step by averaging its window
    step_metrics = []
    for step_idx, step in enumerate(step_events):
        f_idx = step['frame_idx']
        leg = step['leg']
        window_start = max(0, f_idx - 2)
        window_end = min(len(frames_data), f_idx + 3)
        
        leans, flexions, ratios, dists = [], [], [], []
        
        for i in range(window_start, window_end):
            if i < len(frames_data):
                m = frames_data[i]['metrics'].get(leg, {})
                if m.get('valid', False):
                    leans.append(m['torso_lean'])
                    flexions.append(m['knee_flexion'])
                    ratios.append(m['overstride_ratio'])
                    dists.append(m['overstride_distance'])
                
        if len(ratios) > 0:
            step_data = {
                'step_number': step_idx + 1,
                'frame_idx': f_idx,
                'leg': leg,
                'torso_lean': float(np.mean(leans)),
                'knee_flexion': float(np.mean(flexions)),
                'overstride_ratio': float(np.mean(ratios)),
                'overstride_distance': float(np.mean(dists)),
                'is_overstriding': bool(np.mean(ratios) > 0.08),
                'window_start': window_start,
                'window_end': window_end
            }
            step_metrics.append(step_data)
            
    # Compute overall Averages
    if len(step_metrics) > 0:
        overall_torso = float(np.mean([s['torso_lean'] for s in step_metrics]))
        overall_knee = float(np.mean([s['knee_flexion'] for s in step_metrics]))
        overall_overstride = float(np.mean([s['overstride_ratio'] for s in step_metrics]))
        is_overstriding = bool(overall_overstride > 0.08)
    else:
        overall_torso = overall_knee = overall_overstride = 0.0
        is_overstriding = False
            
    import time
    run_id = int(time.time())
    
    left_dir = f"debug_images/run_{run_id}/left_tracking"
    right_dir = f"debug_images/run_{run_id}/right_tracking"
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    
    frame_to_step = {}
    for step in step_metrics:
        # A frame might belong to multiple windows if steps are crazy fast, map to the latest step
        for i in range(step['window_start'], step['window_end']):
            frame_to_step[i] = step
            
    def draw_leg_metrics(m, frame_to_draw):
        hip_pt = (int(m['hip'][0]), int(m['hip'][1]))
        knee_pt = (int(m['knee'][0]), int(m['knee'][1]))
        ankle_pt = (int(m['ankle'][0]), int(m['ankle'][1]))
        
        v_hip = (int(m['virtual_hip'][0]), int(m['virtual_hip'][1]))
        v_shoulder = (int(m['virtual_shoulder'][0]), int(m['virtual_shoulder'][1]))
        
        # 1. Torso Lean (Vertical Line + Virtual Spine Line)
        vertical_top = (v_hip[0], v_hip[1] - 150)
        cv2.line(frame_to_draw, v_hip, vertical_top, (0, 255, 255), 2, cv2.LINE_AA) 
        cv2.line(frame_to_draw, v_hip, v_shoulder, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame_to_draw, f"Lean: {m['torso_lean']:.1f} deg", (v_hip[0] + 20, v_hip[1] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        # 2. Knee Flexion (Hip->Knee, Knee->Ankle)
        cv2.line(frame_to_draw, hip_pt, knee_pt, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.line(frame_to_draw, knee_pt, ankle_pt, (255, 0, 0), 3, cv2.LINE_AA)
        
        # Extend thigh line
        dx_thigh = knee_pt[0] - hip_pt[0]
        dy_thigh = knee_pt[1] - hip_pt[1]
        len_thigh = math.hypot(dx_thigh, dy_thigh)
        len_shin = math.hypot(ankle_pt[0] - knee_pt[0], ankle_pt[1] - knee_pt[1])
        if len_thigh > 0:
            ext_x = int(knee_pt[0] + (dx_thigh / len_thigh) * len_shin)
            ext_y = int(knee_pt[1] + (dy_thigh / len_thigh) * len_shin)
            cv2.line(frame_to_draw, knee_pt, (ext_x, ext_y), (0, 165, 255), 2, cv2.LINE_AA)
            
        cv2.putText(frame_to_draw, f"Knee: {m['knee_flexion']:.1f} deg", (knee_pt[0] + 20, knee_pt[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 3. Overstride (Horizontal line floor)
        floor_y = ankle_pt[1]
        hip_proj = (hip_pt[0], floor_y)
        cv2.line(frame_to_draw, hip_pt, hip_proj, (0, 255, 255), 2, cv2.LINE_AA)
        
        is_frame_overstride = m['overstride_ratio'] > 0.08
        cv2.line(frame_to_draw, hip_proj, ankle_pt, (0, 0, 255) if is_frame_overstride else (0, 255, 0), 3, cv2.LINE_AA)
        
        cv2.putText(frame_to_draw, f"Overstride: {m['overstride_ratio']*100:.1f}% leg", (hip_proj[0], floor_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_frame_overstride else (0, 255, 0), 2)
            
    # Draw overlays
    for i, data in enumerate(frames_data):
        frame_base = data['frame'].copy()
        kpts = data['keypoints']
        
        if kpts is not None:
            # Draw Skeleton lines
            for conn in SKELETON_CONNECTIONS:
                if len(kpts) > max(conn):
                    pt1 = (int(kpts[conn[0]][0]), int(kpts[conn[0]][1]))
                    pt2 = (int(kpts[conn[1]][0]), int(kpts[conn[1]][1]))
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame_base, pt1, pt2, (255, 255, 0), 2)
            
            # Draw Keypoints
            for pt_idx, pt in enumerate(kpts):
                x, y = int(pt[0]), int(pt[1])
                conf = float(pt[2]) if len(pt) > 2 else 0.0
                if x > 0 and y > 0:
                    text_offset = -10
                    if show_keypoint_numbers:
                        cv2.putText(frame_base, str(pt_idx), (x, y + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        text_offset -= 15
                    if show_keypoint_confidence:
                        cv2.putText(frame_base, f"{conf:.2f}", (x, y + text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    if not show_keypoint_numbers:
                        cv2.circle(frame_base, (x, y), 5, (0, 255, 0), -1)
                        
        # 1. Output ALL frames for Left tracking
        m_left = data['metrics'].get('Left', {})
        if m_left.get('valid', False):
            frame_left = frame_base.copy()
            cv2.putText(frame_left, f"LEFT TRACKING - Frame {i}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
            draw_leg_metrics(m_left, frame_left)
            cv2.imwrite(f"{left_dir}/frame_{i:04d}.jpg", frame_left)

        # 2. Output ALL frames for Right tracking
        m_right = data['metrics'].get('Right', {})
        if m_right.get('valid', False):
            frame_right = frame_base.copy()
            cv2.putText(frame_right, f"RIGHT TRACKING - Frame {i}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 3)
            draw_leg_metrics(m_right, frame_right)
            cv2.imwrite(f"{right_dir}/frame_{i:04d}.jpg", frame_right)

        # 3. Create the Main Debug Video using the step windows
        frame_video = frame_base.copy()
        if i in frame_to_step:
            step = frame_to_step[i]
            leg = step['leg']
            m = data['metrics'].get(leg, {})
            
            if m.get('valid', False):
                if i == step['frame_idx']:
                    cv2.putText(frame_video, f"STEP {step['step_number']} ({leg}) STRIKE", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    cv2.putText(frame_video, f"STEP {step['step_number']} ({leg}) WIN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                draw_leg_metrics(m, frame_video)
            
        out.write(frame_video)
        
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
        'debug_run_dir': f"debug_images/run_{run_id}",
        'torso_lean': overall_torso,
        'knee_flexion': overall_knee,
        'overstride_ratio': overall_overstride,
        'is_overstriding': is_overstriding,
        'step_metrics': step_metrics
    }
