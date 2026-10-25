"""
Inverse Kinematics Solver for MediaPipe → GLB Avatar Animation
=============================================================
Reads smoothed MediaPipe landmark JSON, solves IK rotations for a Mixamo
skeleton loaded from a GLB file, and writes the animation back into the GLB.

Key Design Decisions:
- Bones are processed root-to-leaf (topological order) so parent world
  rotations are always computed before children.
- MediaPipe "Left" hand data is routed to the avatar's RIGHT hand bones
  (mirror/selfie swap), and vice versa.
- Coordinate conversion: MediaPipe (x right, y down, z toward cam) →
  glTF (x right, y up, z toward viewer). Transform: (x, -y, -z).
- Rest-pose bone directions are extracted from the child's local translation
  in the GLB, NOT assumed to be Y-axis.
- scipy.spatial.transform.Rotation is used for robust quaternion math.
"""

import json
import struct
import os
import sys
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from pygltflib import GLTF2, Animation, AnimationChannel, \
    AnimationChannelTarget, AnimationSampler, Accessor, BufferView

# ============================================================================
# MediaPipe Pose Landmark Indices (from subject's own perspective)
# ============================================================================
MP_NOSE = 0
MP_LEFT_EYE_INNER = 1
MP_LEFT_EYE = 2
MP_LEFT_EYE_OUTER = 3
MP_RIGHT_EYE_INNER = 4
MP_RIGHT_EYE = 5
MP_RIGHT_EYE_OUTER = 6
MP_LEFT_EAR = 7
MP_RIGHT_EAR = 8
MP_MOUTH_LEFT = 9
MP_MOUTH_RIGHT = 10
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24

# MediaPipe Hand Landmark Indices
MP_HAND_WRIST = 0
MP_HAND_THUMB_CMC = 1
MP_HAND_THUMB_MCP = 2
MP_HAND_THUMB_IP = 3
MP_HAND_THUMB_TIP = 4
MP_HAND_INDEX_MCP = 5
MP_HAND_INDEX_PIP = 6
MP_HAND_INDEX_DIP = 7
MP_HAND_INDEX_TIP = 8
MP_HAND_MIDDLE_MCP = 9
MP_HAND_MIDDLE_PIP = 10
MP_HAND_MIDDLE_DIP = 11
MP_HAND_MIDDLE_TIP = 12
MP_HAND_RING_MCP = 13
MP_HAND_RING_PIP = 14
MP_HAND_RING_DIP = 15
MP_HAND_RING_TIP = 16
MP_HAND_PINKY_MCP = 17
MP_HAND_PINKY_PIP = 18
MP_HAND_PINKY_DIP = 19
MP_HAND_PINKY_TIP = 20


# ============================================================================
# Bone class - represents a single joint in the skeleton
# ============================================================================
class Bone:
    """Represents a single joint/bone in the skeleton hierarchy."""
    
    def __init__(self, node_idx, name, local_translation, local_rotation_quat):
        """
        :param node_idx: glTF node index
        :param name: bone name from the GLB
        :param local_translation: [x, y, z] position relative to parent
        :param local_rotation_quat: [x, y, z, w] quaternion in parent space
        """
        self.node_idx = node_idx
        self.name = name
        self.local_translation = np.array(local_translation, dtype=np.float64)
        # Store as scipy Rotation for easy quaternion math
        self.rest_local_rot = Rotation.from_quat(local_rotation_quat)  # [x,y,z,w]
        
        self.parent = None  # type: Bone | None
        self.children = []  # type: list[Bone]
        
        # Will be set after hierarchy is built
        self._rest_axis = None  # direction this bone points in local space
        
    @property
    def rest_axis(self):
        """
        The direction this bone points toward its child, in bone-local space.
        Computed from the first child's local translation.
        """
        if self._rest_axis is not None:
            return self._rest_axis
        if not self.children:
            return np.array([0, 1, 0], dtype=np.float64)  # default up
        # Use first child's translation direction as the bone axis
        child_t = self.children[0].local_translation
        norm = np.linalg.norm(child_t)
        if norm < 1e-8:
            return np.array([0, 1, 0], dtype=np.float64)
        return child_t / norm
    
    @rest_axis.setter
    def rest_axis(self, value):
        self._rest_axis = value


# ============================================================================
# Skeleton Loader
# ============================================================================
def load_skeleton(gltf):
    """
    Load the skeleton hierarchy from a GLB file.
    Returns a dict {node_idx: Bone} and the root bone.
    """
    if not gltf.skins:
        raise ValueError("GLB file has no skin (no skeleton)")
    
    skin = gltf.skins[0]
    joint_set = set(skin.joints)
    
    bones = {}
    
    # Create Bone objects for every joint
    for node_idx in skin.joints:
        node = gltf.nodes[node_idx]
        t = node.translation if node.translation else [0, 0, 0]
        r = node.rotation if node.rotation else [0, 0, 0, 1]  # identity
        bones[node_idx] = Bone(node_idx, node.name, t, r)
    
    # Build parent-child relationships
    for node_idx in skin.joints:
        node = gltf.nodes[node_idx]
        if node.children:
            for child_idx in node.children:
                if child_idx in bones:
                    bones[child_idx].parent = bones[node_idx]
                    bones[node_idx].children.append(bones[child_idx])
    
    # Find skeleton root
    skeleton_root_idx = skin.skeleton
    if skeleton_root_idx is None:
        # Fallback: find the joint with no parent
        for idx, bone in bones.items():
            if bone.parent is None:
                skeleton_root_idx = idx
                break
    
    root_bone = bones[skeleton_root_idx]
    
    # Set rest axes for multi-child bones that need a specific reference child.
    # For hand bones, use the middle finger (MiddleFinger1) as the direction,
    # since the hand should point along the middle finger axis.
    _set_rest_axis_overrides(bones)
    
    return bones, root_bone


def _set_rest_axis_overrides(bones):
    """
    Override rest_axis for bones that have multiple children and need
    a specific child to determine their direction.
    """
    # Build a name lookup
    name_to_bone = {}
    for bone in bones.values():
        # Strip the numeric suffix for easier matching
        base_name = bone.name.rsplit('_', 1)[0] if '_' in bone.name else bone.name
        name_to_bone[base_name] = bone
        name_to_bone[bone.name] = bone
    
    # For hand bones, override to use middle finger direction
    for hand_name, middle_name in [
        ("LeftHand", "LeftHandMiddle1"),
        ("RightHand", "RightHandMiddle1")
    ]:
        hand_bone = name_to_bone.get(hand_name)
        if hand_bone:
            # Find the Middle1 child
            for child in hand_bone.children:
                child_base = child.name.rsplit('_', 1)[0] if '_' in child.name else child.name
                if child_base == middle_name:
                    ct = child.local_translation
                    norm = np.linalg.norm(ct)
                    if norm > 1e-8:
                        hand_bone.rest_axis = ct / norm
                    break


# ============================================================================
# Name-based bone lookup
# ============================================================================
def find_bone_by_name(bones, prefix):
    """Find a bone whose name starts with the given prefix."""
    for bone in bones.values():
        base = bone.name.rsplit('_', 1)[0] if '_' in bone.name else bone.name
        if base == prefix:
            return bone
    return None


# ============================================================================
# Topological sort (BFS from root)
# ============================================================================
def topological_order(root_bone):
    """Return bones in BFS order from root to leaves."""
    order = []
    queue = [root_bone]
    while queue:
        bone = queue.pop(0)
        order.append(bone)
        queue.extend(bone.children)
    return order


# ============================================================================
# Quaternion Math Utilities
# ============================================================================
def safe_normalize(v):
    """Normalize a vector, returning [0,1,0] if magnitude is near zero."""
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    return v / n


def quat_from_two_vectors(v_from, v_to):
    """
    Compute the shortest-arc rotation that maps v_from to v_to.
    Returns a scipy Rotation object.
    
    Uses the half-angle formula:
      q = (cross + (1 + dot)) normalized
    This avoids gimbal lock and handles edge cases.
    """
    v_from = safe_normalize(v_from)
    v_to = safe_normalize(v_to)
    
    if v_from is None or v_to is None:
        return Rotation.identity()
    
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    
    # Vectors already aligned
    if dot > 0.999999:
        return Rotation.identity()
    
    # Vectors are opposite - pick a perpendicular axis
    if dot < -0.999999:
        # Find a vector not parallel to v_from
        perp = np.array([1, 0, 0], dtype=np.float64)
        if abs(np.dot(v_from, perp)) > 0.9:
            perp = np.array([0, 1, 0], dtype=np.float64)
        axis = np.cross(v_from, perp)
        axis = axis / np.linalg.norm(axis)
        # 180 degree rotation: quaternion is (axis, 0) = [ax, ay, az, 0]
        return Rotation.from_quat([axis[0], axis[1], axis[2], 0.0])
    
    # General case: half-angle formula
    cross = np.cross(v_from, v_to)
    w = 1.0 + dot
    q = np.array([cross[0], cross[1], cross[2], w])
    q = q / np.linalg.norm(q)
    return Rotation.from_quat(q)


# ============================================================================
# Coordinate Space Conversion
# ============================================================================
def mp_to_avatar(lm):
    """
    Convert a MediaPipe landmark {x, y, z} to the skeleton's world space.
    
    MediaPipe: x=right in image, y=down in image, z=closer (negative)
    Skeleton:  x=left/right, y=forward/depth, z=up
    
    The skeleton has the Hips bone with a ~96 deg X rotation that converts
    the internal Y-up bone space to Z-up skeleton world space (FBX artifact).
    So we map to Z-up, not standard glTF Y-up:
      skeleton_X = mp_x   (left/right preserved)
      skeleton_Y = lm['z']  (depth: closer z (negative) maps to negative Y (Forward in Viewer +Z))
      skeleton_Z = -lm['y'] (up: negative image-y maps to positive Z)
    """
    return np.array([lm['x'], lm['z'], -lm['y']], dtype=np.float64)


def direction_from_landmarks(lm_start, lm_end):
    """
    Compute a normalized direction vector from two MediaPipe landmarks.
    Returns None if the direction is zero-length.
    """
    d = mp_to_avatar(lm_end) - mp_to_avatar(lm_start)
    return safe_normalize(d)


# ============================================================================
# IK Bone Mapping
# ============================================================================
def build_ik_mapping(bones):
    """
    Build the mapping from bone names to their IK data source.
    
    Returns a dict: {bone_node_idx: {
        'type': 'pose' | 'hand',
        'source': 'left' | 'right',   # which hand data to read (for hands)
        'start_lm': int,              # landmark index for start of direction
        'end_lm': int,                # landmark index for end of direction
    }}
    
    CRITICAL: MediaPipe hand labels use selfie convention:
    - MediaPipe "Left" hand → person's RIGHT hand → avatar's RIGHT* bones
    - MediaPipe "Right" hand → person's LEFT hand → avatar's LEFT* bones
    """
    mapping = {}
    
    # --- Pose-driven bones ---
    # Spine: direction from hip center to shoulder center
    pose_bones = {
        # Bone prefix: (start_landmark, end_landmark)
        "Spine": (None, None),  # special: hip_center → shoulder_center
        "Neck": (None, None),   # special: shoulder_center → nose
        "Head": (None, None),   # special: nose → eye_center
        
        # Left side of avatar = subject's left side (no swap for pose)
        "LeftShoulder": {'start_lm': MP_LEFT_SHOULDER, 'end_lm': MP_LEFT_SHOULDER},
        "LeftArm": {
            'start_lm': MP_LEFT_SHOULDER, 'end_lm': MP_LEFT_ELBOW,
            'twist_lm': MP_LEFT_WRIST, 'rest_twist_axis': [0.0, 0.0, -1.0] # DOWN
        },
        "LeftForeArm": {'start_lm': MP_LEFT_ELBOW, 'end_lm': MP_LEFT_WRIST},
        
        # Right side of avatar = subject's right side
        "RightShoulder": {'start_lm': MP_RIGHT_SHOULDER, 'end_lm': MP_RIGHT_SHOULDER},
        "RightArm": {
            'start_lm': MP_RIGHT_SHOULDER, 'end_lm': MP_RIGHT_ELBOW,
            'twist_lm': MP_RIGHT_WRIST, 'rest_twist_axis': [0.0, 0.0, 1.0] # UP
        },
        "RightForeArm": {'start_lm': MP_RIGHT_ELBOW, 'end_lm': MP_RIGHT_WRIST},
    }
    
    for prefix, info in pose_bones.items():
        bone = find_bone_by_name(bones, prefix)
        if bone:
            if isinstance(info, tuple):
                entry = {
                    'type': 'pose',
                    'bone_prefix': prefix,
                    'start_lm': info[0],
                    'end_lm': info[1],
                }
            else:
                entry = {
                    'type': 'pose',
                    'bone_prefix': prefix,
                    'start_lm': info.get('start_lm'),
                    'end_lm': info.get('end_lm'),
                }
                if 'twist_lm' in info:
                    entry['twist_lm'] = info['twist_lm']
                    entry['rest_twist_axis'] = np.array(info['rest_twist_axis'], dtype=np.float64)
            mapping[bone.node_idx] = entry
    
    # --- Hand-driven bones ---
    # Remember the label swap: avatar's LEFT hand ← MP "Right" data
    #                          avatar's RIGHT hand ← MP "Left" data
    
    hand_bone_map = [
        # (avatar_bone_prefix, mp_data_source, start_hand_lm, end_hand_lm)
        # Avatar LEFT hand ← MediaPipe "Right" hand data
        ("LeftHand",        "right", MP_HAND_WRIST,      MP_HAND_MIDDLE_MCP),
        ("LeftHandThumb1",  "right", MP_HAND_THUMB_CMC,  MP_HAND_THUMB_MCP),
        ("LeftHandThumb2",  "right", MP_HAND_THUMB_MCP,  MP_HAND_THUMB_IP),
        ("LeftHandThumb3",  "right", MP_HAND_THUMB_IP,   MP_HAND_THUMB_TIP),
        ("LeftHandIndex1",  "right", MP_HAND_INDEX_MCP,  MP_HAND_INDEX_PIP),
        ("LeftHandIndex2",  "right", MP_HAND_INDEX_PIP,  MP_HAND_INDEX_DIP),
        ("LeftHandIndex3",  "right", MP_HAND_INDEX_DIP,  MP_HAND_INDEX_TIP),
        ("LeftHandMiddle1", "right", MP_HAND_MIDDLE_MCP, MP_HAND_MIDDLE_PIP),
        ("LeftHandMiddle2", "right", MP_HAND_MIDDLE_PIP, MP_HAND_MIDDLE_DIP),
        ("LeftHandMiddle3", "right", MP_HAND_MIDDLE_DIP, MP_HAND_MIDDLE_TIP),
        ("LeftHandRing1",   "right", MP_HAND_RING_MCP,   MP_HAND_RING_PIP),
        ("LeftHandRing2",   "right", MP_HAND_RING_PIP,   MP_HAND_RING_DIP),
        ("LeftHandRing3",   "right", MP_HAND_RING_DIP,   MP_HAND_RING_TIP),
        ("LeftHandPinky1",  "right", MP_HAND_PINKY_MCP,  MP_HAND_PINKY_PIP),
        ("LeftHandPinky2",  "right", MP_HAND_PINKY_PIP,  MP_HAND_PINKY_DIP),
        ("LeftHandPinky3",  "right", MP_HAND_PINKY_DIP,  MP_HAND_PINKY_TIP),
        
        # Avatar RIGHT hand ← MediaPipe "Left" hand data
        ("RightHand",        "left", MP_HAND_WRIST,      MP_HAND_MIDDLE_MCP),
        ("RightHandThumb1",  "left", MP_HAND_THUMB_CMC,  MP_HAND_THUMB_MCP),
        ("RightHandThumb2",  "left", MP_HAND_THUMB_MCP,  MP_HAND_THUMB_IP),
        ("RightHandThumb3",  "left", MP_HAND_THUMB_IP,   MP_HAND_THUMB_TIP),
        ("RightHandIndex1",  "left", MP_HAND_INDEX_MCP,  MP_HAND_INDEX_PIP),
        ("RightHandIndex2",  "left", MP_HAND_INDEX_PIP,  MP_HAND_INDEX_DIP),
        ("RightHandIndex3",  "left", MP_HAND_INDEX_DIP,  MP_HAND_INDEX_TIP),
        ("RightHandMiddle1", "left", MP_HAND_MIDDLE_MCP, MP_HAND_MIDDLE_PIP),
        ("RightHandMiddle2", "left", MP_HAND_MIDDLE_PIP, MP_HAND_MIDDLE_DIP),
        ("RightHandMiddle3", "left", MP_HAND_MIDDLE_DIP, MP_HAND_MIDDLE_TIP),
        ("RightHandRing1",   "left", MP_HAND_RING_MCP,   MP_HAND_RING_PIP),
        ("RightHandRing2",   "left", MP_HAND_RING_PIP,   MP_HAND_RING_DIP),
        ("RightHandRing3",   "left", MP_HAND_RING_DIP,   MP_HAND_RING_TIP),
        ("RightHandPinky1",  "left", MP_HAND_PINKY_MCP,  MP_HAND_PINKY_PIP),
        ("RightHandPinky2",  "left", MP_HAND_PINKY_PIP,  MP_HAND_PINKY_DIP),
        ("RightHandPinky3",  "left", MP_HAND_PINKY_DIP,  MP_HAND_PINKY_TIP),
    ]
    
    for bone_prefix, mp_source, start_lm, end_lm in hand_bone_map:
        bone = find_bone_by_name(bones, bone_prefix)
        if bone:
            mapping[bone.node_idx] = {
                'type': 'hand',
                'source': mp_source,  # which JSON key to read
                'bone_prefix': bone_prefix,
                'start_lm': start_lm,
                'end_lm': end_lm,
            }
    
    return mapping


# ============================================================================
# Target Direction Extraction
# ============================================================================
def get_target_direction(ik_info, frame_data):
    """
    Given a bone's IK mapping info and a frame of MediaPipe data,
    compute the target direction vector in avatar world space.
    Returns None if data is missing for this frame.
    """
    if ik_info['type'] == 'pose':
        pose = frame_data.get('pose_landmarks', [])
        if len(pose) < 25:  # need at least up to hip landmarks
            return None
        
        prefix = ik_info['bone_prefix']
        
        # Special computed directions
        if prefix == "Spine":
            # Hip center → Shoulder center
            hip_center = {
                'x': (pose[MP_LEFT_HIP]['x'] + pose[MP_RIGHT_HIP]['x']) / 2,
                'y': (pose[MP_LEFT_HIP]['y'] + pose[MP_RIGHT_HIP]['y']) / 2,
                'z': (pose[MP_LEFT_HIP]['z'] + pose[MP_RIGHT_HIP]['z']) / 2,
            }
            shoulder_center = {
                'x': (pose[MP_LEFT_SHOULDER]['x'] + pose[MP_RIGHT_SHOULDER]['x']) / 2,
                'y': (pose[MP_LEFT_SHOULDER]['y'] + pose[MP_RIGHT_SHOULDER]['y']) / 2,
                'z': (pose[MP_LEFT_SHOULDER]['z'] + pose[MP_RIGHT_SHOULDER]['z']) / 2,
            }
            return direction_from_landmarks(hip_center, shoulder_center)
        
        elif prefix == "Neck":
            # Shoulder center → Ear center
            # Using ears instead of nose because the nose is far forward of the
            # skeletal neck axis, creating a ~58 deg forward lean. The ears are
            # much more aligned with the actual head-on-spine direction (~29 deg).
            shoulder_center = {
                'x': (pose[MP_LEFT_SHOULDER]['x'] + pose[MP_RIGHT_SHOULDER]['x']) / 2,
                'y': (pose[MP_LEFT_SHOULDER]['y'] + pose[MP_RIGHT_SHOULDER]['y']) / 2,
                'z': (pose[MP_LEFT_SHOULDER]['z'] + pose[MP_RIGHT_SHOULDER]['z']) / 2,
            }
            ear_center = {
                'x': (pose[MP_LEFT_EAR]['x'] + pose[MP_RIGHT_EAR]['x']) / 2,
                'y': (pose[MP_LEFT_EAR]['y'] + pose[MP_RIGHT_EAR]['y']) / 2,
                'z': (pose[MP_LEFT_EAR]['z'] + pose[MP_RIGHT_EAR]['z']) / 2,
            }
            return direction_from_landmarks(shoulder_center, ear_center)
        
        elif prefix == "Head":
            # Nose → midpoint between eyes
            eye_center = {
                'x': (pose[MP_LEFT_EYE]['x'] + pose[MP_RIGHT_EYE]['x']) / 2,
                'y': (pose[MP_LEFT_EYE]['y'] + pose[MP_RIGHT_EYE]['y']) / 2,
                'z': (pose[MP_LEFT_EYE]['z'] + pose[MP_RIGHT_EYE]['z']) / 2,
            }
            return direction_from_landmarks(pose[MP_NOSE], eye_center)
        
        elif prefix in ("LeftShoulder", "RightShoulder"):
            # Clavicle: from spine center outward to shoulder
            shoulder_center = {
                'x': (pose[MP_LEFT_SHOULDER]['x'] + pose[MP_RIGHT_SHOULDER]['x']) / 2,
                'y': (pose[MP_LEFT_SHOULDER]['y'] + pose[MP_RIGHT_SHOULDER]['y']) / 2,
                'z': (pose[MP_LEFT_SHOULDER]['z'] + pose[MP_RIGHT_SHOULDER]['z']) / 2,
            }
            if prefix == "LeftShoulder":
                return direction_from_landmarks(shoulder_center, pose[MP_LEFT_SHOULDER])
            else:
                return direction_from_landmarks(shoulder_center, pose[MP_RIGHT_SHOULDER])
        
        else:
            # Standard: direction from start_lm to end_lm
            start_lm = ik_info['start_lm']
            end_lm = ik_info['end_lm']
            if start_lm is None or end_lm is None:
                return None
            return direction_from_landmarks(pose[start_lm], pose[end_lm])
    
    elif ik_info['type'] == 'hand':
        # Which JSON key to read
        source = ik_info['source']
        key = f"{source}_hand_landmarks"
        hand_data = frame_data.get(key, [])
        
        if len(hand_data) < 21:
            return None
        
        start_lm = ik_info['start_lm']
        end_lm = ik_info['end_lm']
        return direction_from_landmarks(hand_data[start_lm], hand_data[end_lm])
    
    return None


def get_target_normal(ik_info, frame_data):
    """
    Compute the plane normal (twist target) for bones with a twist_lm.
    """
    if 'twist_lm' not in ik_info:
        return None
        
    pose = frame_data.get('pose_landmarks', [])
    if len(pose) < 25:
        return None
        
    start_lm = ik_info['start_lm']
    end_lm = ik_info['end_lm']
    twist_lm = ik_info['twist_lm']
    
    if start_lm is None or end_lm is None or twist_lm is None:
        return None
        
    p1 = mp_to_avatar(pose[start_lm])
    p2 = mp_to_avatar(pose[end_lm])
    p3 = mp_to_avatar(pose[twist_lm])
    
    v1 = p2 - p1
    v2 = p3 - p2
    
    normal = np.cross(v1, v2)
    return safe_normalize(normal)


# ============================================================================
# IK Solver Core
# ============================================================================
def solve_frame(topo_order, ik_mapping, frame_data):
    """
    Solve IK for a single frame using DELTA rotations on top of the rest pose.
    
    For each bone in topological order (root → leaf):
    1. Compute the "hypothetical rest world" = parent_current_world * rest_local_rot.
       This is what the bone's world rotation WOULD be if it kept its rest rotation,
       given the parent's current (possibly IK-modified) world rotation.
    2. If the bone has an IK target:
       a. Compute rest_world_dir = rest_world.apply(rest_axis)
       b. Compute delta = quat_from_two_vectors(rest_world_dir, target_dir)
       c. Apply delta ON TOP: new_world = delta * rest_world
       d. Extract new_local = inv(parent_world) * new_world
    3. This preserves the bone's rest-pose twist and FBX→glTF correction rotations,
       while only adjusting the pointing direction.
    
    Returns a dict: {node_idx: [x, y, z, w] quaternion}
    """
    # Stores the accumulated world rotation for each bone
    world_rots = {}  # node_idx → Rotation
    # Stores the computed local rotation for each bone
    local_rots = {}  # node_idx → Rotation
    
    for bone in topo_order:
        # Get parent's CURRENT world rotation (includes IK modifications from parents)
        if bone.parent is not None:
            parent_world = world_rots[bone.parent.node_idx]
        else:
            parent_world = Rotation.identity()
        
        # Compute "hypothetical rest world" — what this bone's world rotation
        # WOULD be if it kept its rest rotation, given parent's current state.
        rest_world = parent_world * bone.rest_local_rot
        
        # Check if this bone has an IK target
        target_dir = None
        target_normal = None
        if bone.node_idx in ik_mapping:
            target_dir = get_target_direction(ik_mapping[bone.node_idx], frame_data)
            target_normal = get_target_normal(ik_mapping[bone.node_idx], frame_data)
        
        if target_dir is not None:
            # Compute rest-pose world direction
            rest_world_dir = rest_world.apply(bone.rest_axis)
            rest_world_dir_n = safe_normalize(rest_world_dir)
            
            if rest_world_dir_n is not None:
                # Compute world-space delta: rotation from rest direction to target
                delta_world = quat_from_two_vectors(rest_world_dir_n, target_dir)
                
                # Apply twist correction if a target normal is provided
                if target_normal is not None and 'rest_twist_axis' in ik_mapping[bone.node_idx]:
                    rest_twist = ik_mapping[bone.node_idx]['rest_twist_axis']
                    # Where does the twist axis point after the primary alignment?
                    cur_twist = delta_world.apply(rest_twist)
                    
                    # Project both current and target twist onto the plane perpendicular to target_dir
                    def project_on_plane(v, normal):
                        return v - np.dot(v, normal) * normal
                    
                    cur_twist_proj = safe_normalize(project_on_plane(cur_twist, target_dir))
                    target_normal_proj = safe_normalize(project_on_plane(target_normal, target_dir))
                    
                    if cur_twist_proj is not None and target_normal_proj is not None:
                        twist_rot = quat_from_two_vectors(cur_twist_proj, target_normal_proj)
                        delta_world = twist_rot * delta_world
                
                # Apply delta ON TOP of rest world rotation (preserves twist)
                new_world = delta_world * rest_world
                # Extract new local rotation: new_local = inv(parent) * new_world
                new_local = parent_world.inv() * new_world
                
                local_rots[bone.node_idx] = new_local
                world_rots[bone.node_idx] = new_world
            else:
                # Fallback to rest pose
                local_rots[bone.node_idx] = bone.rest_local_rot
                world_rots[bone.node_idx] = rest_world
        else:
            # No IK target: use rest pose rotation (but with updated parent chain)
            local_rots[bone.node_idx] = bone.rest_local_rot
            world_rots[bone.node_idx] = rest_world
    
    # Convert to quaternion arrays [x, y, z, w]
    result = {}
    for node_idx, rot in local_rots.items():
        if node_idx in ik_mapping:
            result[node_idx] = rot.as_quat()  # [x, y, z, w]
    
    return result


# ============================================================================
# Animation Writer (Binary GLB Packing)
# ============================================================================
def write_animation(gltf, animated_node_indices, all_frame_rotations, timestamps_sec):
    """
    Write the solved IK rotations as a glTF animation into the GLB.
    
    :param gltf: GLTF2 object (will be modified in-place)
    :param animated_node_indices: list of node indices that have animation data
    :param all_frame_rotations: list of dicts, one per frame.
                                Each dict: {node_idx: [x,y,z,w]}
    :param timestamps_sec: list of floats, timestamp in seconds for each frame
    """
    num_frames = len(timestamps_sec)
    num_bones = len(animated_node_indices)
    
    if num_frames == 0 or num_bones == 0:
        print("WARNING: No animation data to write.")
        return
    
    print(f"Writing animation: {num_frames} frames × {num_bones} bones")
    
    # Get the existing binary blob
    blob = gltf.binary_blob()
    if blob is None:
        blob = b""
    existing_buffer_length = len(blob)
    
    # Ensure the buffer is 4-byte aligned before adding new data
    padding = (4 - (existing_buffer_length % 4)) % 4
    blob += b'\x00' * padding
    new_data_offset = existing_buffer_length + padding
    
    # --- Pack timestamps ---
    # One SCALAR accessor for all timestamps
    timestamps_bytes = struct.pack(f'<{num_frames}f', *timestamps_sec)
    ts_offset = new_data_offset
    ts_byte_length = len(timestamps_bytes)
    
    # --- Pack quaternions for each bone ---
    # One VEC4 accessor per bone
    bone_data_list = []  # list of (byte_offset, byte_data, node_idx)
    current_offset = ts_offset + ts_byte_length
    
    # Pad to 4-byte boundary after timestamps
    ts_padding = (4 - (current_offset % 4)) % 4
    current_offset += ts_padding
    
    for node_idx in animated_node_indices:
        quats = []
        for frame_idx in range(num_frames):
            frame_rots = all_frame_rotations[frame_idx]
            if node_idx in frame_rots:
                q = frame_rots[node_idx]
            else:
                # Fallback: identity quaternion
                q = [0, 0, 0, 1]
            quats.extend(q)  # x, y, z, w
        
        quat_bytes = struct.pack(f'<{num_frames * 4}f', *quats)
        bone_data_list.append((current_offset, quat_bytes, node_idx))
        current_offset += len(quat_bytes)
    
    # --- Assemble the binary blob ---
    new_binary = timestamps_bytes
    new_binary += b'\x00' * ts_padding  # padding after timestamps
    for _, quat_bytes, _ in bone_data_list:
        new_binary += quat_bytes
    
    # Append to the existing blob
    full_blob = blob + new_binary
    
    # Update the buffer length
    if not gltf.buffers:
        raise ValueError("GLB has no buffers")
    gltf.buffers[0].byteLength = len(full_blob)
    
    # --- Create BufferViews and Accessors ---
    # BufferView for timestamps
    ts_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=ts_offset,
        byteLength=ts_byte_length,
    ))
    
    # Accessor for timestamps
    ts_acc_idx = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=ts_bv_idx,
        byteOffset=0,
        componentType=5126,  # FLOAT
        count=num_frames,
        type="SCALAR",
        min=[timestamps_sec[0]],
        max=[timestamps_sec[-1]],
    ))
    
    # BufferViews and Accessors for each bone's quaternion data
    samplers = []
    channels = []
    
    for i, (byte_offset, quat_bytes, node_idx) in enumerate(bone_data_list):
        bv_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=byte_offset,
            byteLength=len(quat_bytes),
        ))
        
        acc_idx = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=bv_idx,
            byteOffset=0,
            componentType=5126,  # FLOAT
            count=num_frames,
            type="VEC4",
        ))
        
        # Sampler: input = timestamps, output = quaternions
        sampler_idx = len(samplers)
        samplers.append(AnimationSampler(
            input=ts_acc_idx,
            output=acc_idx,
            interpolation="LINEAR",
        ))
        
        # Channel: target this bone's rotation
        channels.append(AnimationChannel(
            sampler=sampler_idx,
            target=AnimationChannelTarget(
                node=node_idx,
                path="rotation",
            ),
        ))
    
    # Create the animation
    animation = Animation(
        name="MediaPipe_IK",
        samplers=samplers,
        channels=channels,
    )
    
    if gltf.animations is None:
        gltf.animations = []
    gltf.animations.append(animation)
    
    # Set the updated binary blob
    gltf.set_binary_blob(full_blob)
    
    print(f"Animation written: {len(channels)} channels, "
          f"{len(full_blob)} bytes total buffer")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Main IK solver pipeline."""
    
    # --- Configuration ---
    INPUT_DIR = "output"
    ASSETS_DIR = "assets"
    GLB_INPUT = os.path.join(ASSETS_DIR, "avatar.glb")
    
    # Find the smoothed JSON file
    import glob
    smooth_files = glob.glob(os.path.join(INPUT_DIR, "*_smooth.json"))
    if not smooth_files:
        print("ERROR: No smoothed JSON files found in output/")
        print("Run smoother.py first.")
        sys.exit(1)
    
    json_path = smooth_files[0]
    base_name = os.path.basename(json_path).replace("features_", "").replace("_smooth.json", "")
    output_glb = os.path.join(INPUT_DIR, f"animated_{base_name}.glb")
    
    print(f"=== IK Solver ===")
    print(f"Input JSON: {json_path}")
    print(f"Input GLB:  {GLB_INPUT}")
    print(f"Output GLB: {output_glb}")
    print()
    
    # --- Step 1: Load the GLB skeleton ---
    print("[1/5] Loading skeleton from GLB...")
    gltf = GLTF2.load(GLB_INPUT)
    bones, root_bone = load_skeleton(gltf)
    topo_order = topological_order(root_bone)
    print(f"  Loaded {len(bones)} bones. Root: {root_bone.name}")
    
    # --- Step 2: Build IK mapping ---
    print("[2/5] Building IK bone mapping...")
    ik_mapping = build_ik_mapping(bones)
    print(f"  Mapped {len(ik_mapping)} bones for IK")
    for node_idx, info in sorted(ik_mapping.items()):
        bone = bones[node_idx]
        print(f"    [{node_idx:2d}] {bone.name:35s} <- {info['type']:5s} "
              f"lm {info['start_lm']} -> {info['end_lm']}")
    
    # --- Step 3: Load JSON data ---
    print("[3/5] Loading smoothed landmark data...")
    with open(json_path, 'r') as f:
        frames_data = json.load(f)
    print(f"  {len(frames_data)} frames loaded")
    
    # --- Step 4: Solve IK for each frame ---
    print("[4/5] Solving IK rotations...")
    all_frame_rotations = []
    timestamps_sec = []
    animated_node_set = set()
    
    for i, frame in enumerate(frames_data):
        ts = frame['timestamp_ms'] / 1000.0
        timestamps_sec.append(ts)
        
        frame_rots = solve_frame(topo_order, ik_mapping, frame)
        all_frame_rotations.append(frame_rots)
        animated_node_set.update(frame_rots.keys())
        
        if i % 20 == 0:
            print(f"  Frame {i}/{len(frames_data)}: "
                  f"{len(frame_rots)} bones animated")
    
    # Sorted list of animated node indices
    animated_node_indices = sorted(animated_node_set)
    print(f"  Total unique animated bones: {len(animated_node_indices)}")
    
    # --- Step 5: Write animation into GLB ---
    print("[5/5] Writing animation to GLB...")
    
    # Reload the GLB fresh to avoid any state issues
    gltf_out = GLTF2.load(GLB_INPUT)
    write_animation(gltf_out, animated_node_indices, all_frame_rotations, timestamps_sec)
    
    gltf_out.save(output_glb)
    print(f"\n[OK] Animated GLB saved to: {output_glb}")
    print(f"  Duration: {timestamps_sec[-1]:.2f}s, "
          f"Frames: {len(timestamps_sec)}, "
          f"Bones: {len(animated_node_indices)}")


if __name__ == "__main__":
    main()
