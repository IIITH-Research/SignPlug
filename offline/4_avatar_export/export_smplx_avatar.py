# export_smplx_avatar.py — Export a rigged SMPL-X avatar as GLB for three.js.
#
# Run this INSIDE your existing `smplerx` conda env (same one that runs
# interpolate_and_render_word.py), since it needs `smplx` + `torch` + your
# licensed human_model_files.
#
#   python export_smplx_avatar.py
#
# Output: ./male2k_smplx.glb  — load this in three.js instead of male2k.glb
#
# DESIGN DECISION: the exported skeleton uses IDENTITY bind-pose rotations
# (every bone's rest rotation = identity quaternion). This means your existing
# three.js line:
#     bone.quaternion.copy(restQuaternions[bname].clone().multiply(qAnim))
# reduces to bone.quaternion.copy(qAnim) — no axis-frame mismatch to debug.
# Bone names below match SMPLX_BODY_BONES / SMPLX_LHAND_BONES / SMPLX_RHAND_BONES
# in live_interpolator.py EXACTLY. If you rename one, rename the other.

import os
import json
import numpy as np
import torch
import smplx
import pygltflib
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView,
    Accessor, Skin, Asset
)

MODEL_PATH = '/ssd_scratch/sharif/ISL-AVATAR-PROJECT/skinning/SMPLer-X/common/utils/human_model_files'
OUT_GLB = 'male2k_smplx.glb'
GENDER = 'male'

# --- Joint order: MUST match live_interpolator.py's SMPLX_* lists, in the
# exact index order the smplx package uses internally (parents array order).
BONE_NAMES = (
    ['Pelvis',
     'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
     'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
     'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
     'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist']            # 0-21  (22 joints)
    + ['Jaw', 'L_Eye', 'R_Eye']                              # 22-24 (3 joints, not driven)
    + ['L_Index1', 'L_Index2', 'L_Index3',
       'L_Middle1', 'L_Middle2', 'L_Middle3',
       'L_Pinky1', 'L_Pinky2', 'L_Pinky3',
       'L_Ring1', 'L_Ring2', 'L_Ring3',
       'L_Thumb1', 'L_Thumb2', 'L_Thumb3']                  # 25-39 (15 joints)
    + ['R_Index1', 'R_Index2', 'R_Index3',
       'R_Middle1', 'R_Middle2', 'R_Middle3',
       'R_Pinky1', 'R_Pinky2', 'R_Pinky3',
       'R_Ring1', 'R_Ring2', 'R_Ring3',
       'R_Thumb1', 'R_Thumb2', 'R_Thumb3']                  # 40-54 (15 joints)
)
NUM_JOINTS = len(BONE_NAMES)  # 55


def build_model():
    model = smplx.create(
        MODEL_PATH, 'smplx', gender=GENDER, num_betas=10,
        use_face_contour=True, flat_hand_mean=False, use_pca=False, batch_size=1
    )
    return model


def get_rest_pose(model):
    """Zero-pose (T-pose) forward pass -> vertices, joints, skin weights, parents."""
    with torch.no_grad():
        out = model(
            betas=torch.zeros((1, 10)),
            body_pose=torch.zeros((1, 21, 3)),
            left_hand_pose=torch.zeros((1, 15, 3)),
            right_hand_pose=torch.zeros((1, 15, 3)),
            jaw_pose=torch.zeros((1, 3)),
            global_orient=torch.zeros((1, 3)),
            return_verts=True,
        )
    verts = out.vertices[0].cpu().numpy().astype(np.float32)         # (V, 3)
    joints_full = out.joints[0].cpu().numpy().astype(np.float32)     # (J_full, 3)
    joints = joints_full[:NUM_JOINTS]                                # keep only main 55

    parents = model.parents.cpu().numpy()[:NUM_JOINTS]               # (55,)
    weights_full = model.lbs_weights.detach().cpu().numpy()          # (V, J_model)
    weights = weights_full[:, :NUM_JOINTS].astype(np.float32)        # (V, 55)

    print(f"verts={verts.shape} joints_full={joints_full.shape} "
          f"using_joints={joints.shape} parents={parents.shape} weights={weights.shape}")
    assert joints.shape[0] == NUM_JOINTS, (
        f"Expected {NUM_JOINTS} joints, got {joints.shape[0]} — your smplx version's "
        f"joint order may differ. Print model.parents and cross-check before proceeding."
    )
    return verts, joints, parents, weights


def top4_skin_weights(weights):
    """glTF skin needs vec4 JOINTS_0/WEIGHTS_0 per vertex -> keep top-4 influences, renormalize."""
    V, J = weights.shape
    joints_idx = np.zeros((V, 4), dtype=np.uint16)
    joints_w = np.zeros((V, 4), dtype=np.float32)
    top4 = np.argsort(-weights, axis=1)[:, :4]
    for v in range(V):
        idx = top4[v]
        w = weights[v, idx]
        s = w.sum()
        if s > 1e-8:
            w = w / s
        joints_idx[v] = idx
        joints_w[v] = w
    return joints_idx, joints_w


def export_glb(verts, joints, parents, weights):
    joints_idx, joints_w = top4_skin_weights(weights)

    # glTF Y-up vs SMPL-X: SMPL-X is already roughly Y-up (standing pose), so no axis swap needed.
    faces = None
    # smplx exposes faces via model.faces (numpy array) after a forward pass
    from_model_faces_path = None  # placeholder, set below after model() call in main()

    # --- binary buffer layout ---
    buffers_bytes = bytearray()

    def add_block(arr):
        nonlocal buffers_bytes
        offset = len(buffers_bytes)
        data = arr.tobytes()
        buffers_bytes += data
        # pad to 4-byte alignment
        pad = (-len(buffers_bytes)) % 4
        buffers_bytes += b'\x00' * pad
        return offset, len(data)

    gltf = GLTF2(asset=Asset(version="2.0", generator="export_smplx_avatar.py"))

    # POSITION
    pos_off, pos_len = add_block(verts)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=pos_off, byteLength=pos_len, target=34962))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=5126, count=len(verts),
        type="VEC3", min=verts.min(axis=0).tolist(), max=verts.max(axis=0).tolist()
    ))
    pos_accessor = len(gltf.accessors) - 1

    # JOINTS_0
    j_off, j_len = add_block(joints_idx)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=j_off, byteLength=j_len, target=34962))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=5123, count=len(joints_idx), type="VEC4"
    ))
    joints_accessor = len(gltf.accessors) - 1

    # WEIGHTS_0
    w_off, w_len = add_block(joints_w)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=w_off, byteLength=w_len, target=34962))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=5126, count=len(joints_w), type="VEC4"
    ))
    weights_accessor = len(gltf.accessors) - 1

    # INDICES (faces) — filled in by caller before this function via global FACES
    faces_arr = FACES.astype(np.uint32)
    idx_off, idx_len = add_block(faces_arr)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=idx_off, byteLength=idx_len, target=34963))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=5125, count=faces_arr.size, type="SCALAR"
    ))
    indices_accessor = len(gltf.accessors) - 1

    # inverseBindMatrices: since bind rotation = identity, inverse bind = translate(-joint_pos)
    ibms = np.zeros((NUM_JOINTS, 4, 4), dtype=np.float32)
    for j in range(NUM_JOINTS):
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -joints[j]
        # FIX: glTF requires column-major flattening. numpy's .tobytes() writes
        # row-major, so store the TRANSPOSE here -> tobytes() then emits the
        # correct column-major byte order. Without this, every translation
        # component lands in the wrong slot in the buffer, which corrupts the
        # skin and is why the viewer showed a black screen with 165 errors
        # (confirmed with the official Khronos gltf-validator: ACCESSOR_INVALID_IBM).
        ibms[j] = m.T
    ibm_off, ibm_len = add_block(ibms)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=ibm_off, byteLength=ibm_len))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=5126, count=NUM_JOINTS, type="MAT4"
    ))
    ibm_accessor = len(gltf.accessors) - 1

    # --- joint nodes: local translation = joint_pos - parent_joint_pos, identity rotation ---
    joint_node_indices = []
    for j in range(NUM_JOINTS):
        if parents[j] < 0:
            local_t = joints[j]
        else:
            local_t = joints[j] - joints[parents[j]]
        node = Node(name=BONE_NAMES[j], translation=local_t.tolist())
        gltf.nodes.append(node)
        joint_node_indices.append(len(gltf.nodes) - 1)

    # wire up children (must be done after all nodes exist since indices are needed)
    for j in range(NUM_JOINTS):
        if parents[j] >= 0:
            parent_node = gltf.nodes[joint_node_indices[parents[j]]]
            if parent_node.children is None:
                parent_node.children = []
            parent_node.children.append(joint_node_indices[j])

    root_joint_node = joint_node_indices[0]  # Pelvis

    # --- mesh node (skinned) ---
    mesh = Mesh(primitives=[Primitive(
        attributes=Attributes(POSITION=pos_accessor, JOINTS_0=joints_accessor, WEIGHTS_0=weights_accessor),
        indices=indices_accessor
    )])
    gltf.meshes.append(mesh)

    skin = Skin(inverseBindMatrices=ibm_accessor, joints=joint_node_indices, skeleton=root_joint_node)
    gltf.skins.append(skin)

    mesh_node = Node(name="SMPLX_Mesh", mesh=0, skin=0)
    gltf.nodes.append(mesh_node)
    mesh_node_idx = len(gltf.nodes) - 1

    gltf.scenes.append(Scene(nodes=[root_joint_node, mesh_node_idx]))
    gltf.scene = 0

    gltf.buffers.append(Buffer(byteLength=len(buffers_bytes)))
    gltf.set_binary_blob(bytes(buffers_bytes))

    gltf.save(OUT_GLB)
    print(f"Saved {OUT_GLB}  ({len(verts)} verts, {NUM_JOINTS} joints)")


FACES = None

if __name__ == "__main__":
    model = build_model()
    FACES = np.array(model.faces)  # (F, 3) triangle indices, from the loaded model
    verts, joints, parents, weights = get_rest_pose(model)
    export_glb(verts, joints, parents, weights)
