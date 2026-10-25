import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// SCENE SETUP
// ============================================================================
const scene = new THREE.Scene();
const container = document.getElementById('canvas-wrapper') || document.getElementById('avatar-container') || document.body;
const width = container.clientWidth || window.innerWidth;
const height = container.clientHeight || window.innerHeight;

const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
camera.position.set(0, 1.0, 3); // Closer for SMPL-X

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();

window.addEventListener('resize', () => {
    const container = document.getElementById('canvas-wrapper') || document.body;
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;
    
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
});

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(5, 5, 5);
scene.add(dirLight);

// Add a ground plane and grid
const gridHelper = new THREE.GridHelper(10, 10, 0x888888, 0x444444);
gridHelper.position.y = -1.0;
scene.add(gridHelper);

const platformGeo = new THREE.CylinderGeometry(1.5, 1.5, 0.05, 32);
const platformMat = new THREE.MeshStandardMaterial({ color: 0x333333, roughness: 0.8, metalness: 0.2 });
const platform = new THREE.Mesh(platformGeo, platformMat);
platform.position.y = -1.025; // Flush with grid
scene.add(platform);

// ============================================================================
// BONE TABLES
// ============================================================================

const SMPLX_BONE_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 
    'left_eye_smplhf', 'right_eye_smplhf', 
    'left_index1', 'left_index2', 'left_index3', 
    'left_middle1', 'left_middle2', 'left_middle3', 
    'left_pinky1', 'left_pinky2', 'left_pinky3', 
    'left_ring1', 'left_ring2', 'left_ring3', 
    'left_thumb1', 'left_thumb2', 'left_thumb3', 
    'right_index1', 'right_index2', 'right_index3', 
    'right_middle1', 'right_middle2', 'right_middle3', 
    'right_pinky1', 'right_pinky2', 'right_pinky3', 
    'right_ring1', 'right_ring2', 'right_ring3', 
    'right_thumb1', 'right_thumb2', 'right_thumb3'
];

// Display-friendly names for the sidebar
const smplxDisplayNames = {};
SMPLX_BONE_NAMES.forEach(name => {
    smplxDisplayNames[name] = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
});

// ALL 55 bones must be tracked so rotations from the JSON actually get applied
const targetBones = [...SMPLX_BONE_NAMES];

// Bones the user can click and rotate with the gizmo (upper body only)
const interactableBones = [
    'pelvis', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'jaw',
    'left_collar', 'right_collar',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_eye_smplhf', 'right_eye_smplhf',
    'left_index1', 'left_index2', 'left_index3',
    'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3',
    'left_ring1', 'left_ring2', 'left_ring3',
    'left_thumb1', 'left_thumb2', 'left_thumb3',
    'right_index1', 'right_index2', 'right_index3',
    'right_middle1', 'right_middle2', 'right_middle3',
    'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_foot', 'right_foot'
];

// Parent -> primary child for drawing bone cones (using semantic names)
const primaryChild = {
    'pelvis': 'spine1',
    'spine1': 'spine2',
    'spine2': 'spine3',
    'spine3': 'neck',
    'neck': 'head',
    'left_collar': 'left_shoulder',
    'right_collar': 'right_shoulder',
    'left_shoulder': 'left_elbow',
    'right_shoulder': 'right_elbow',
    'left_elbow': 'left_wrist',
    'right_elbow': 'right_wrist',
    'left_wrist': 'left_middle1',
    'right_wrist': 'right_middle1',
    'left_index1': 'left_index2', 'left_index2': 'left_index3',
    'left_middle1': 'left_middle2', 'left_middle2': 'left_middle3',
    'left_pinky1': 'left_pinky2', 'left_pinky2': 'left_pinky3',
    'left_ring1': 'left_ring2', 'left_ring2': 'left_ring3',
    'left_thumb1': 'left_thumb2', 'left_thumb2': 'left_thumb3',
    'right_index1': 'right_index2', 'right_index2': 'right_index3',
    'right_middle1': 'right_middle2', 'right_middle2': 'right_middle3',
    'right_pinky1': 'right_pinky2', 'right_pinky2': 'right_pinky3',
    'right_ring1': 'right_ring2', 'right_ring2': 'right_ring3',
    'right_thumb1': 'right_thumb2', 'right_thumb2': 'right_thumb3',
    'left_hip': 'left_knee', 'right_hip': 'right_knee',
    'left_knee': 'left_ankle', 'right_knee': 'right_ankle',
    'left_ankle': 'left_foot', 'right_ankle': 'right_foot',
};

// ============================================================================
// AVATAR CONFIGURATIONS
// ============================================================================
// Each avatar config maps SMPLX's 55 bone indices to the actual bone names in the GLB.
// boneMapping[i] = the name of the bone in this GLB that corresponds to SMPLX_BONE_NAMES[i].
// null means the GLB has no matching bone for that SMPLX joint.

const AVATAR_CONFIGS = {
    smplx: {
        name: 'SMPL-X Default',
        glbPath: './assets/smplx_male.glb',
        scale: [1.0, 1.0, 1.0],
        position: [0, 0, 0],
        needsRetarget: false, // SMPLX GLB has identity rest rotations, no retargeting needed
        makeTransparent: true, // SMPLX is a plain mesh, make it semi-transparent to see bones
        meshNameFilter: (name) => name === 'SMPLX_Mesh' || name.includes('Mesh'),
        // boneMapping: identity â€” bone names in GLB match SMPLX_BONE_NAMES exactly
        boneMapping: [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
            'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
            'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw',
            'left_eye_smplhf', 'right_eye_smplhf',
            'left_index1', 'left_index2', 'left_index3',
            'left_middle1', 'left_middle2', 'left_middle3',
            'left_pinky1', 'left_pinky2', 'left_pinky3',
            'left_ring1', 'left_ring2', 'left_ring3',
            'left_thumb1', 'left_thumb2', 'left_thumb3',
            'right_index1', 'right_index2', 'right_index3',
            'right_middle1', 'right_middle2', 'right_middle3',
            'right_pinky1', 'right_pinky2', 'right_pinky3',
            'right_ring1', 'right_ring2', 'right_ring3',
            'right_thumb1', 'right_thumb2', 'right_thumb3'
        ]
    },
    maleavatar: {
        name: 'Male Avatar (Realistic)',
        glbPath: './assets/maleavatar.glb',
        scale: [1.0, 1.0, 1.0],
        mapRotation: (poseQ, boneName) => {
            const q = new THREE.Quaternion().copy(poseQ);
            const isLeftArm = ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandIndex', 'LeftHandMiddle', 'LeftHandRing', 'LeftHandPinky', 'LeftHandThumb'].some(n => boneName.includes(n));
            const isRightArm = ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex', 'RightHandMiddle', 'RightHandRing', 'RightHandPinky', 'RightHandThumb'].some(n => boneName.includes(n));
            
            if (isLeftArm) {
                // Mixamo Left Arm: World X->Y, Y->Z, Z->X
                q.set(poseQ.z, poseQ.x, poseQ.y, poseQ.w);
            } else if (isRightArm) {
                // Mixamo Right Arm: World X->-Y, Y->-Z, Z->X
                q.set(poseQ.z, -poseQ.x, -poseQ.y, poseQ.w);
            }
            return q;
        },
        unmapRotation: (boneQ, boneName) => {
            const q = new THREE.Quaternion().copy(boneQ);
            const isLeftArm = ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandIndex', 'LeftHandMiddle', 'LeftHandRing', 'LeftHandPinky', 'LeftHandThumb'].some(n => boneName.includes(n));
            const isRightArm = ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex', 'RightHandMiddle', 'RightHandRing', 'RightHandPinky', 'RightHandThumb'].some(n => boneName.includes(n));
            
            if (isLeftArm) {
                // Inverse of X->Y, Y->Z, Z->X is X->Z, Y->X, Z->Y
                q.set(boneQ.y, boneQ.z, boneQ.x, boneQ.w);
            } else if (isRightArm) {
                // Inverse of X->-Y, Y->-Z, Z->X is X->Z, Y->-X, Z->-Y
                q.set(-boneQ.y, -boneQ.z, boneQ.x, boneQ.w);
            }
            return q;
        },
        fingerCorrection: (poseQ, boneName) => {
            // Uncurl the fingers by -15 degrees on Z axis (which is the curl axis in Mixamo after swizzle)
            const uncurl = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -15 * Math.PI / 180));
            return poseQ.clone().multiply(uncurl);
        },
        position: [0, 0, 0],
        needsRetarget: true, // Mixamo skeleton has non-identity rest rotations
        makeTransparent: false, // Textured model â€” keep original materials
        meshNameFilter: (name) => true, // Accept all meshes

        boneMapping: [
            // 0-11: body
            'Hips_01', 'LeftUpLeg_057', 'RightUpLeg_062', 'Spine_02',
            'LeftLeg_058', 'RightLeg_063', 'Spine1_03',
            'LeftFoot_059', 'RightFoot_064', 'Spine2_04',
            'LeftToeBase_060', 'RightToeBase_065', 'Neck_05',
            // 13-21: upper body
            'LeftShoulder_010', 'RightShoulder_034', 'Head_06',
            'LeftArm_011', 'RightArm_035',
            'LeftForeArm_012', 'RightForeArm_036',
            'LeftHand_013', 'RightHand_037',
            // 22-24: face
            null, // jaw (no jaw bone in MaleAvatar)
            'LeftEye_08', 'RightEye_09',
            // 25-39: left hand fingers (SMPLX order: index, middle, pinky, ring, thumb)
            'LeftHandIndex1_018', 'LeftHandIndex2_019', 'LeftHandIndex3_020',
            'LeftHandMiddle1_022', 'LeftHandMiddle2_023', 'LeftHandMiddle3_024',
            'LeftHandPinky1_030', 'LeftHandPinky2_031', 'LeftHandPinky3_032',
            'LeftHandRing1_026', 'LeftHandRing2_027', 'LeftHandRing3_028',
            'LeftHandThumb1_014', 'LeftHandThumb2_015', 'LeftHandThumb3_016',
            // 40-54: right hand fingers (SMPLX order: index, middle, pinky, ring, thumb)
            'RightHandIndex1_042', 'RightHandIndex2_043', 'RightHandIndex3_044',
            'RightHandMiddle1_046', 'RightHandMiddle2_047', 'RightHandMiddle3_048',
            'RightHandPinky1_054', 'RightHandPinky2_055', 'RightHandPinky3_056',
            'RightHandRing1_050', 'RightHandRing2_051', 'RightHandRing3_052',
            'RightHandThumb1_038', 'RightHandThumb2_039', 'RightHandThumb3_040'
        ]
    }
};

let activeAvatarKey = 'smplx';
let currentAvatarScene = null; // Reference to the loaded avatar scene object

const boneObjects       = {};
const restQuaternions   = {};
const restPositions     = {};
const targetQuaternions = {};
const restWorldQuaternions = {}; // Stores world quaternions of all bones in rest pose

let frameBackupRotations = null;

let boneHitMeshes = [];    // Cylinder meshes for picking bones
let ringMeshes = [];       // Torus ring meshes for rotation gizmo
let selectedBoneName = null;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// Rotation gizmo group (3 torus rings)
let rotationGizmo = null;

// Rotation ring drag state
let isRotDragging = false;
let rotDragAxis = null;         // 'x', 'y', or 'z'
let rotDragStartAngle = 0;
let rotDragStartQuat = new THREE.Quaternion();
let rotDragBoneWorldPos = new THREE.Vector3();

let isManualMode = false;
let manualFrames = [];
let hasUnsavedChanges = false;
function setUnsavedChanges(val) {
    hasUnsavedChanges = val;
    const discardBtn = document.getElementById('discard-btn');
    if (discardBtn) {
        discardBtn.style.display = val ? 'block' : 'none';
    }
}
let modelLoaded = false;
let avatarMeshes = [];

// ============================================================================
// MODEL LOAD
// ============================================================================
const loader = new GLTFLoader();

/**
 * Load an avatar by key. Cleans up the previous avatar, loads the new GLB,
 * builds bone mappings, hit meshes, and gizmo.
 * All bones are stored in boneObjects keyed by their SMPLX semantic name
 * so that all downstream code (applyFrameData, persistBoneChange, etc.)
 * works identically regardless of which avatar is loaded.
 */
function loadAvatar(avatarKey) {
    const config = AVATAR_CONFIGS[avatarKey];
    if (!config) { console.error('Unknown avatar:', avatarKey); return; }

    // â”€â”€ Cleanup previous avatar â”€â”€
    modelLoaded = false;
    selectedBoneName = null;
    if (rotationGizmo) { rotationGizmo.visible = false; }

    // Remove old hit meshes from their parent bones (they were added as children)
    boneHitMeshes.forEach(m => { if (m.parent) m.parent.remove(m); });
    boneHitMeshes = [];

    // Clear bone dicts
    for (const k of Object.keys(boneObjects)) delete boneObjects[k];
    for (const k of Object.keys(restQuaternions)) delete restQuaternions[k];
    for (const k of Object.keys(restPositions)) delete restPositions[k];
    for (const k of Object.keys(targetQuaternions)) delete targetQuaternions[k];
    avatarMeshes = [];

    // Remove old avatar scene from Three.js scene
    if (currentAvatarScene) {
        scene.remove(currentAvatarScene);
        currentAvatarScene.traverse(child => {
            if (child.isMesh) {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
                    else child.material.dispose();
                }
            }
        });
        currentAvatarScene = null;
    }

    // Reset grid/platform position
    gridHelper.position.y = -1.0;
    platform.position.y = -1.025;

    activeAvatarKey = avatarKey;

    // â”€â”€ Build reverse mapping: GLB bone name â†’ SMPLX semantic name â”€â”€
    const glbNameToSmplx = {};
    config.boneMapping.forEach((glbName, smplxIdx) => {
        if (glbName) glbNameToSmplx[glbName] = SMPLX_BONE_NAMES[smplxIdx];
    });

    // â”€â”€ Also build a reverse primaryChild map using GLB names â”€â”€
    // For non-SMPLX avatars, child bones have different names, so we need to
    // look up the GLB child bone name from the primaryChild map via SMPLX names.
    const glbPrimaryChild = {};
    for (const [parentSmplx, childSmplx] of Object.entries(primaryChild)) {
        const parentIdx = SMPLX_BONE_NAMES.indexOf(parentSmplx);
        const childIdx = SMPLX_BONE_NAMES.indexOf(childSmplx);
        if (parentIdx >= 0 && childIdx >= 0) {
            const parentGlb = config.boneMapping[parentIdx];
            const childGlb = config.boneMapping[childIdx];
            if (parentGlb && childGlb) {
                glbPrimaryChild[parentGlb] = childGlb;
            }
        }
    }
    loader.load(config.glbPath + '?v=' + Date.now(), (gltf) => {
        const avatar = gltf.scene;
        scene.add(avatar);
        currentAvatarScene = avatar;

        avatar.scale.set(...config.scale);
        avatar.position.set(...config.position);
        avatar.updateMatrixWorld(true);

        avatar.traverse((child) => {
            if (child.isMesh) {
                child.frustumCulled = false;
                if (config.meshNameFilter(child.name)) {
                    avatarMeshes.push(child);
                    if (config.makeTransparent) {
                        // SMPLX: make semi-transparent to see bones through the mesh
                        const mats = Array.isArray(child.material) ? child.material : [child.material];
                        mats.forEach(mat => {
                            mat.transparent = true;
                            mat.opacity = 0.5;
                        });
                    }
                    // For textured models, keep original materials untouched
                }
            }
            if (!child.isBone) return;

            // Look up the SMPLX semantic name for this bone
            const smplxName = glbNameToSmplx[child.name];
            if (!smplxName) return; // This bone has no SMPLX equivalent, skip it

            boneObjects[smplxName]       = child;
            restQuaternions[smplxName]   = child.quaternion.clone();
            restPositions[smplxName]     = child.position.clone();
            targetQuaternions[smplxName] = child.quaternion.clone();
            
            // Store world rest quaternions for ALL bones (handled below in a separate traversal)

            if (!interactableBones.includes(smplxName)) return;

            // Find the primary child bone using GLB names
            const glbName = child.name;
            let childBone = null;
            if (glbPrimaryChild[glbName]) {
                childBone = child.children.find(c => c.name === glbPrimaryChild[glbName]);
                // For Mixamo models, the child might be nested deeper (e.g., through twist bones)
                if (!childBone) {
                    child.traverse(descendant => {
                        if (descendant.name === glbPrimaryChild[glbName]) childBone = descendant;
                    });
                }
            }

            let length = 0.05;
            let boneVec = new THREE.Vector3(0, 1, 0);

            if (childBone && childBone.position.length() > 0.0001) {
                length = childBone.position.length();
                boneVec = childBone.position.clone().normalize();
            } else if (child.parent && child.parent.isBone) {
                const parentWorldPos = new THREE.Vector3();
                child.parent.getWorldPosition(parentWorldPos);
                const childWorldPos = new THREE.Vector3();
                child.getWorldPosition(childWorldPos);
                const dirWorld = new THREE.Vector3().subVectors(childWorldPos, parentWorldPos);
                length = child.position.length();
                const invQuat = new THREE.Quaternion();
                child.getWorldQuaternion(invQuat).invert();
                boneVec = dirWorld.applyQuaternion(invQuat).normalize();
                if (boneVec.lengthSq() < 0.001) boneVec.set(0, 1, 0);
            }

            if (smplxName === 'left_foot' || smplxName === 'right_foot') {
                length = 0.12;
                boneVec.set(0, 0, 1).normalize();
                const worldPos = new THREE.Vector3();
                child.getWorldPosition(worldPos);
                if (gridHelper.position.y === -1.0 || worldPos.y - 0.05 < gridHelper.position.y) {
                    gridHelper.position.y = worldPos.y - 0.02;
                    platform.position.y = worldPos.y - 0.045;
                }
            }

            const isFinger = smplxName.includes('index') || smplxName.includes('middle') || smplxName.includes('pinky') || smplxName.includes('ring') || smplxName.includes('thumb');
            let radius = isFinger ? 0.006 : 0.015;

            const hitGeo = new THREE.CylinderGeometry(radius, radius, length, 8);
            hitGeo.rotateX(Math.PI / 2);
            hitGeo.translate(0, 0, length / 2);
            const hitMat = new THREE.MeshBasicMaterial({ visible: false, depthTest: false });
            const hitMesh = new THREE.Mesh(hitGeo, hitMat);

            const isEye = smplxName.includes('eye');
            const isHead = smplxName === 'head' || smplxName === 'jaw' || smplxName.includes('neck');
            let visualRadius = isFinger ? 0.004 : 0.015;
            let visualLength = length;
            if (isEye) { visualRadius = 0.006; visualLength = Math.min(length, 0.03); }
            else if (isHead) { visualRadius = 0.012; visualLength = length * 0.8; }

            const w = visualRadius;
            const w_z = Math.min(visualLength * 0.2, 0.02);
            const vertices = new Float32Array([
                0, 0, visualLength, 0, 0, 0,
                w, 0, w_z, -w, 0, w_z, 0, w, w_z, 0, -w, w_z
            ]);
            const indices = [
                0, 2, 4,  0, 4, 3,  0, 3, 5,  0, 5, 2,
                1, 4, 2,  1, 3, 4,  1, 5, 3,  1, 2, 5
            ];
            const boneGeo = new THREE.BufferGeometry();
            boneGeo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            boneGeo.setIndex(indices);
            boneGeo.computeVertexNormals();
            const boneMat = new THREE.MeshBasicMaterial({
                color: 0x88ccff, opacity: 0.6, transparent: true, depthTest: false
            });
            const visibleBone = new THREE.Mesh(boneGeo, boneMat);

            hitMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), boneVec);
            visibleBone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), boneVec);

            // Store the SMPLX semantic name for picking
            hitMesh.userData.boneName = smplxName;
            hitMesh.userData.visibleBone = visibleBone;
            child.add(hitMesh);
            child.add(visibleBone);
            boneHitMeshes.push(hitMesh);
        });

        avatar.updateMatrixWorld(true);

        // â”€â”€ Build gizmo (only once) â”€â”€
        if (!rotationGizmo) {
            rotationGizmo = new THREE.Object3D();
            rotationGizmo.visible = false;
            rotationGizmo.renderOrder = 1000;

            const ringRadius = 0.05, tubeRadius = 0.005, ringSegments = 48, tubeSegments = 12;
            const axisColors = { x: 0xff4444, y: 0x44ff44, z: 0x4488ff };
            const axisRotations = {
                x: new THREE.Euler(0, Math.PI / 2, 0),
                y: new THREE.Euler(Math.PI / 2, 0, 0),
                z: new THREE.Euler(0, 0, 0)
            };

            ['x', 'y', 'z'].forEach(axis => {
                const torusGeo = new THREE.TorusGeometry(ringRadius, tubeRadius, tubeSegments, ringSegments);
                const torusMat = new THREE.MeshBasicMaterial({
                    color: axisColors[axis], depthTest: false, transparent: true,
                    opacity: 0.85, side: THREE.DoubleSide
                });
                const ring = new THREE.Mesh(torusGeo, torusMat);
                ring.rotation.copy(axisRotations[axis]);
                ring.userData.axis = axis;
                ring.userData.isGizmoRing = true;
                ring.renderOrder = 1001;
                rotationGizmo.add(ring);
                ringMeshes.push(ring);
            });

            scene.add(rotationGizmo);
        }

        modelLoaded = true;
        console.log(`Avatar "${config.name}" loaded with ${Object.keys(boneObjects).length} mapped bones.`);

        // â”€â”€ Auto-scale and Auto-position â”€â”€
        const bbox = new THREE.Box3();
        const computeBBox = () => {
            bbox.makeEmpty();
            avatarMeshes.forEach(mesh => {
                mesh.updateMatrixWorld(true);
                bbox.union(new THREE.Box3().setFromObject(mesh));
            });
        };
        computeBBox();
        
        // Precompute world rest quaternions for all bones
        avatar.traverse(child => {
            if (child.isBone) {
                const wq = new THREE.Quaternion();
                child.getWorldQuaternion(wq);
                restWorldQuaternions[child.uuid] = wq;
            }
        });
        
        if (!bbox.isEmpty()) {
            let minY = bbox.min.y;
            let maxY = bbox.max.y;
            let height = maxY - minY;
            
            // Auto-scale if the avatar is unusually large (e.g. exported in cm instead of m)
            if (height > 3.0) {
                const scaleFactor = 1.8 / height; // Target height ~1.8m
                avatar.scale.set(scaleFactor, scaleFactor, scaleFactor);
                avatar.updateMatrixWorld(true);
                computeBBox(); // Recompute bounding box after scale
                minY = bbox.min.y;
                maxY = bbox.max.y;
                height = maxY - minY;
                console.log(`Auto-scaled "${config.name}" by ${scaleFactor.toFixed(4)}x to height ${height.toFixed(2)}m`);
            }
            
            const targetFloorY = -1.0;
            const offsetY = targetFloorY - minY;
            avatar.position.y += offsetY;
            
            gridHelper.position.y = targetFloorY - 0.02;
            platform.position.y = targetFloorY - 0.045;
            
            avatar.updateMatrixWorld(true);
            
            console.log(`Auto-positioned "${config.name}": height=${height.toFixed(2)}m, offset=${offsetY.toFixed(3)}`);
        }

        // â”€â”€ Specular-Glossiness Texture Fallback (for George Clooney) â”€â”€
        if (gltf.parser && gltf.parser.json && gltf.parser.json.extensionsUsed && gltf.parser.json.extensionsUsed.includes('KHR_materials_pbrSpecularGlossiness')) {
            const parser = gltf.parser;
            Promise.all(parser.json.meshes.map((_, i) => parser.getDependency('mesh', i).catch(() => null))).then(meshes => {
                parser.json.meshes.forEach((meshDef, meshIndex) => {
                    const threeMeshGroup = meshes[meshIndex];
                    if (!threeMeshGroup) return;
                    
                    const primitives = meshDef.primitives;
                    const threeMeshes = threeMeshGroup.isGroup ? threeMeshGroup.children : [threeMeshGroup];
                    
                    primitives.forEach((primDef, primIndex) => {
                        const matIndex = primDef.material;
                        if (matIndex === undefined) return;
                        
                        const matDef = parser.json.materials[matIndex];
                        if (matDef && matDef.extensions && matDef.extensions.KHR_materials_pbrSpecularGlossiness) {
                            const sgExt = matDef.extensions.KHR_materials_pbrSpecularGlossiness;
                            if (sgExt.diffuseTexture !== undefined) {
                                parser.getDependency('texture', sgExt.diffuseTexture.index).then(texture => {
                                    const threeMesh = threeMeshes[primIndex] || threeMeshes[0];
                                    if (threeMesh && threeMesh.material) {
                                        texture.colorSpace = THREE.SRGBColorSpace;
                                        if (Array.isArray(threeMesh.material)) {
                                            threeMesh.material.forEach(m => { m.map = texture; m.needsUpdate = true; });
                                        } else {
                                            threeMesh.material.map = texture;
                                            threeMesh.material.needsUpdate = true;
                                        }
                                    }
                                }).catch(err => console.error("Texture load error:", err));
                            }
                        }
                    });
                });
            });
        }

        // If we have animation data loaded, re-apply current frame
        if (isDataLoaded && animationFrames.length > 0) {
            applyFrameData(animationFrames[currentFrame]);
        }
        loadData();
    }, undefined, (err) => console.error('GLB load error:', err));
}

// â”€â”€ Avatar selector event â”€â”€
const avatarSelector = document.getElementById('avatar-selector');
if (avatarSelector) {
    avatarSelector.addEventListener('change', (e) => {
        loadAvatar(e.target.value);
    });
}

// â”€â”€ Initial load â”€â”€
loadAvatar('smplx');


// ============================================================================
// ANIMATION & PLAYBACK
// ============================================================================
let animationFrames = [];
let originalAnimationFrames = [];
let currentFrame = 0;
let isDataLoaded = false;
let isPlaying = false;

let activeEuler = new THREE.Euler();
let lastSelectedBoneForEuler = null;
let pendingFrameChange = null;

function updateUIForSelectedBone(fromManualEdit = false) {
    const nameEl = document.getElementById('selected-bone-name');
    if (!nameEl) return; 
    
    if (!selectedBoneName) {
        nameEl.textContent = 'None';
        nameEl.style.color = '#ccc';
        return;
    }
    
    const bone = boneObjects[selectedBoneName];
    if (!bone) return;
    
    nameEl.textContent = smplxDisplayNames[selectedBoneName] || selectedBoneName;
    nameEl.style.color = '#4CAF50';
    
    if (!fromManualEdit || lastSelectedBoneForEuler !== selectedBoneName) {
        activeEuler.setFromQuaternion(bone.quaternion, 'XYZ');
        lastSelectedBoneForEuler = selectedBoneName;
    }
    
    document.getElementById('rot-x-val').value = (activeEuler.x * 180 / Math.PI).toFixed(1);
    document.getElementById('rot-y-val').value = (activeEuler.y * 180 / Math.PI).toFixed(1);
    document.getElementById('rot-z-val').value = (activeEuler.z * 180 / Math.PI).toFixed(1);
}

function updateTranslationUI() {
    if (!animationFrames || animationFrames.length === 0) return;
    const frameData = animationFrames[currentFrame];
    if (frameData && frameData.transl) {
        document.getElementById('trans-x-val').value = frameData.transl[0].toFixed(3);
        document.getElementById('trans-y-val').value = frameData.transl[1].toFixed(3);
        document.getElementById('trans-z-val').value = frameData.transl[2].toFixed(3);
    } else {
        document.getElementById('trans-x-val').value = 0;
        document.getElementById('trans-y-val').value = 0;
        document.getElementById('trans-z-val').value = 0;
    }
}

function showUnsavedModal(targetFrame) {
    pendingFrameChange = targetFrame;
    const modal = document.getElementById('unsaved-modal');
    if (modal) modal.style.display = 'block';
}

function goToFrame(frameIdx) {
    if (!animationFrames || animationFrames.length === 0) return;
    
    if (hasUnsavedChanges) {
        showUnsavedModal(frameIdx);
        return;
    }

    currentFrame = Math.max(0, Math.min(frameIdx, animationFrames.length - 1));
    const slider = document.getElementById('frame-slider');
    if (slider) slider.value = currentFrame;
    
    const currDisp = document.getElementById('current-frame-display');
    const totalDisp = document.getElementById('total-frames-display');
    const frameObj = animationFrames[currentFrame];
    
    if (currDisp) {
        currDisp.textContent = frameObj.index !== undefined ? frameObj.index : currentFrame;
    }
    if (totalDisp) {
        totalDisp.textContent = `(Keyframe ${currentFrame + 1} of ${animationFrames.length})`;
    }
    
    // Update reference image
    const imgEl = document.getElementById('reference-image');
    const noImgEl = document.getElementById('no-image-text');
    if (imgEl && noImgEl) {
        if (frameObj && frameObj.imageUrl) {
            imgEl.src = frameObj.imageUrl;
            imgEl.style.display = 'block';
            noImgEl.style.display = 'none';
        } else {
            imgEl.style.display = 'none';
            noImgEl.style.display = 'block';
        }
    }
    
    // Backup the pristine state of this frame before any edits are made, for the Discard function
    frameBackupRotations = JSON.parse(JSON.stringify(animationFrames[currentFrame].rotations));
    
    deselectBone();
    
    isPlaying = false;
    
    applyFrameData(animationFrames[currentFrame]);
    updateUIForSelectedBone();
    updateTranslationUI();
}

function applyFrameData(frameData) {
    if (!frameData || !frameData.rotations) return;

    const config = AVATAR_CONFIGS[activeAvatarKey];
    const needsRetarget = config && config.needsRetarget;
    
    // Apply global translation relative to the rest position of the root bone
    if (boneObjects[SMPLX_BONE_NAMES[0]]) {
        const rootBoneName = SMPLX_BONE_NAMES[0];
        const bone = boneObjects[rootBoneName];
        const restPos = restPositions[rootBoneName];
        
        if (frameData.transl) {
            bone.position.set(
                restPos.x + frameData.transl[0], 
                restPos.y + frameData.transl[1], 
                restPos.z + frameData.transl[2]
            );
        } else {
            bone.position.copy(restPos);
        }
    }
    
    const SMPLX_ANATOMY = {
        right: { primary: new THREE.Vector3(-1,0,0), bend: new THREE.Vector3(0,1,0), twist: new THREE.Vector3(0,0,-1) },
        left: { primary: new THREE.Vector3(1,0,0), bend: new THREE.Vector3(0,-1,0), twist: new THREE.Vector3(0,0,-1) }
    };
    const MIXAMO_ANATOMY = {
        right: { primary: new THREE.Vector3(0,1,0), bend: new THREE.Vector3(0,0,1), twist: new THREE.Vector3(1,0,0) },
        left:  { primary: new THREE.Vector3(0,1,0), bend: new THREE.Vector3(0,0,1), twist: new THREE.Vector3(1,0,0) }
    };

    const smplxWorldQ = {};
    const glbNameToSmplxRev = {};
    for (let i = 0; i < SMPLX_BONE_NAMES.length; i++) {
        const bn = SMPLX_BONE_NAMES[i];
        if (boneObjects[bn]) glbNameToSmplxRev[boneObjects[bn].name] = bn;
    }

    const getMappedParentWorldQ = (bone) => {
        let curr = bone.parent;
        while (curr && curr.isBone) {
            const smplxName = glbNameToSmplxRev[curr.name];
            if (smplxName && smplxWorldQ[smplxName]) {
                return smplxWorldQ[smplxName].clone();
            }
            curr = curr.parent;
        }
        // If we reach the root of the bones, return the avatar's world orientation
        const rootQ = new THREE.Quaternion();
        if (currentAvatarScene) currentAvatarScene.getWorldQuaternion(rootQ);
        return rootQ;
    };

    // Pass 1: Compute SMPLX Absolute World Quaternions
    for (let i = 0; i < 55; i++) {
        const boneName = SMPLX_BONE_NAMES[i];
        if (!boneName || !frameData.rotations[i]) continue;
        const bone = boneObjects[boneName];
        if (!bone) continue;
        
        const rot = frameData.rotations[i];
        const localQ = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]);
        const parentWorldQ = getMappedParentWorldQ(bone);
        smplxWorldQ[boneName] = parentWorldQ.multiply(localQ);
    }

    // Pass 2: Apply
    for (let i = 0; i < 55; i++) {
        const boneName = SMPLX_BONE_NAMES[i];
        if (!boneName || !frameData.rotations[i]) continue;
        const bone = boneObjects[boneName];
        if (!bone) continue;

        if (needsRetarget && restQuaternions[boneName]) {
            const isLeftArm = ['LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandIndex', 'LeftHandMiddle', 'LeftHandRing', 'LeftHandPinky', 'LeftHandThumb'].some(n => bone.name.includes(n));
            const isRightArm = ['RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandIndex', 'RightHandMiddle', 'RightHandRing', 'RightHandPinky', 'RightHandThumb'].some(n => bone.name.includes(n));
            
            if (isLeftArm || isRightArm) {
                const side = isLeftArm ? 'left' : 'right';
                const sAnat = SMPLX_ANATOMY[side];
                const mAnat = MIXAMO_ANATOMY[side];
                
                const targetPrimary = sAnat.primary.clone().applyQuaternion(smplxWorldQ[boneName]);
                const targetBend = sAnat.bend.clone().applyQuaternion(smplxWorldQ[boneName]);
                const targetTwist = sAnat.twist.clone().applyQuaternion(smplxWorldQ[boneName]);
                
                const mBasis = new THREE.Matrix4().makeBasis(mAnat.primary, mAnat.bend, mAnat.twist);
                const tBasis = new THREE.Matrix4().makeBasis(targetPrimary, targetBend, targetTwist);
                
                const mBasisInv = new THREE.Matrix4().copy(mBasis).invert();
                const wAvatarMat = new THREE.Matrix4().multiplyMatrices(tBasis, mBasisInv);
                const wAvatarQ = new THREE.Quaternion().setFromRotationMatrix(wAvatarMat);
                
                const parentWorldQ = new THREE.Quaternion();
                if (bone.parent) {
                    bone.parent.getWorldQuaternion(parentWorldQ);
                }
                
                const qLocal = parentWorldQ.invert().multiply(wAvatarQ);
                
                const isFinger = bone.name.includes('Index') || bone.name.includes('Middle') || bone.name.includes('Ring') || bone.name.includes('Pinky') || bone.name.includes('Thumb');
                if (isFinger && AVATAR_CONFIGS[activeAvatarKey].fingerCorrection) {
                    AVATAR_CONFIGS[activeAvatarKey].fingerCorrection(qLocal, boneName);
                }
                
                bone.quaternion.copy(qLocal);
                bone.updateMatrixWorld(true);
            } else {
                // Spine / Head / Legs
                const rot = frameData.rotations[i];
                const poseQ = new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]);
                bone.quaternion.copy(restQuaternions[boneName]).multiply(poseQ);
                bone.updateMatrixWorld(true);
            }
        } else {
            const rot = frameData.rotations[i];
            bone.quaternion.copy(new THREE.Quaternion(rot[0], rot[1], rot[2], rot[3]));
            bone.updateMatrixWorld(true);
        }
        targetQuaternions[boneName].copy(bone.quaternion);
    }
    
    if (boneObjects[SMPLX_BONE_NAMES[0]]) boneObjects[SMPLX_BONE_NAMES[0]].updateMatrixWorld(true);

}

async function loadData() {
    // Initial load is empty now until user selects a gloss
}

// ============================================================================
// GLOSS SELECTION API LOGIC
// ============================================================================
const loadGlossBtn = document.getElementById('load-folder-btn');
const glossModal = document.getElementById('gloss-modal');
const closeGlossModal = document.getElementById('close-gloss-modal');
const glossSearch = document.getElementById('gloss-search');
const glossList = document.getElementById('gloss-list');
const tabRaw = document.getElementById('tab-raw');
const tabCorrected = document.getElementById('tab-corrected');

let allGlosses = [];
let activeGlossFolder = "";
const glossDataCache = {};
let activeCategory = "raw"; 

async function fetchGlosses(category) {
    activeCategory = category;
    
    // Update Tab UI
    if (tabRaw && tabCorrected) {
        tabRaw.style.background = category === 'raw' ? '#e91e63' : '#333';
        tabRaw.style.color = category === 'raw' ? 'white' : '#aaa';
        tabCorrected.style.background = category === 'corrected' ? '#e91e63' : '#333';
        tabCorrected.style.color = category === 'corrected' ? 'white' : '#aaa';
    }
    
    glossList.innerHTML = '<div style="padding:15px; color:#aaa;">Loading glosses...</div>';
    try {
        const res = await fetch(`/api/glosses?category=${category}`);
        const data = await res.json();
        allGlosses = data.glosses || []; // Now expects array of objects: {name: "school", has_corrected: true}
        if (glossSearch) glossSearch.value = '';
        renderGlossList(allGlosses);
    } catch (e) {
        glossList.innerHTML = `<div style="padding:15px; color:#f44336;">Failed to load glosses from server. Make sure server.py is running.</div>`;
    }
}

if (tabRaw) tabRaw.addEventListener('click', () => fetchGlosses('raw'));
if (tabCorrected) tabCorrected.addEventListener('click', () => fetchGlosses('corrected'));

if (loadGlossBtn) {
    loadGlossBtn.addEventListener('click', () => {
        glossModal.style.display = 'flex';
        fetchGlosses(activeCategory);
    });
}

if (closeGlossModal) {
    closeGlossModal.addEventListener('click', () => {
        glossModal.style.display = 'none';
    });
}

if (glossSearch) {
    glossSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const filtered = allGlosses.filter(g => g.name.toLowerCase().includes(query));
        renderGlossList(filtered);
    });
}

function renderGlossList(glosses) {
    const progressEl = document.getElementById('progress-counter');
    if (progressEl) {
        if (activeCategory === 'raw' && allGlosses.length > 0) {
            const correctedCount = allGlosses.filter(g => g.has_corrected).length;
            progressEl.textContent = `${correctedCount} / ${allGlosses.length} Done`;
            progressEl.style.display = 'inline-block';
        } else {
            progressEl.style.display = 'none';
        }
    }

    let displayGlosses = [...glosses];
    if (activeCategory === 'raw') {
        displayGlosses.sort((a, b) => {
            if (a.has_corrected === b.has_corrected) {
                return a.name.localeCompare(b.name);
            }
            return a.has_corrected ? 1 : -1;
        });
    } else {
        displayGlosses.sort((a, b) => a.name.localeCompare(b.name));
    }

    if (displayGlosses.length === 0) {
        glossList.innerHTML = '<div style="padding:15px; color:#aaa;">No glosses found.</div>';
        return;
    }
    glossList.innerHTML = '';
    displayGlosses.forEach(glossObj => {
        const gloss = glossObj.name;
        const hasCorrected = glossObj.has_corrected;
        
        const div = document.createElement('div');
        div.innerHTML = `<span>${gloss}</span>` + (hasCorrected ? `<span style="float: right; color: #4CAF50;">✅</span>` : ``);
        div.style.padding = '12px 15px';
        div.style.borderBottom = '1px solid #333';
        div.style.cursor = 'pointer';
        div.style.color = '#ddd';
        
        div.addEventListener('mouseenter', () => div.style.background = '#333');
        div.addEventListener('mouseleave', () => div.style.background = 'transparent');
        
        div.addEventListener('click', () => {
            glossModal.style.display = 'none';
            loadGlossData(gloss, activeCategory);
        });
        glossList.appendChild(div);
    });
}
function prefetchImages(imagesDict) {
    // Fire-and-forget: warms the browser cache so goToFrame's imgEl.src hits instantly
    Object.values(imagesDict).forEach(url => {
        if (!url) return;
        const img = new Image();
        img.src = url;
    });
}
async function loadGlossData(folder, category) {
    activeGlossFolder = folder;
    activeCategory = category;
    const activeFileEl = document.getElementById('active-file-display');
    if (activeFileEl) {
        activeFileEl.innerHTML = `${folder} <span style="font-size:11px; font-weight:normal; padding:2px 6px; border-radius:3px; background:#4CAF50; color:white; margin-left:8px; vertical-align:middle;">Loading...</span>`;
    }

    const cacheKey = `${category}:${folder}`;

    // Reuse already-parsed data instead of re-fetching/re-unzipping
    if (glossDataCache[cacheKey]) {
        animationFrames = JSON.parse(JSON.stringify(glossDataCache[cacheKey].animationFrames));
        
	originalAnimationFrames = JSON.parse(JSON.stringify(glossDataCache[cacheKey].originalAnimationFrames));
	prefetchImages(Object.fromEntries(animationFrames.map(f => [f.index, f.imageUrl])));
        isDataLoaded = true;

        const slider = document.getElementById('frame-slider');
        const totalDisplay = document.getElementById('total-frames-display');
        if (slider) { slider.max = animationFrames.length - 1; slider.value = 0; }
        if (totalDisplay) totalDisplay.textContent = `(Keyframe 1 of ${animationFrames.length})`;

        if (activeFileEl) {
            activeFileEl.innerHTML = `${folder} <span style="font-size:11px; font-weight:normal; padding:2px 6px; border-radius:3px; background:#4CAF50; color:white; margin-left:8px; vertical-align:middle;">SMPL-X NPZ</span>`;
        }

        goToFrame(0);
        return;
    }

    try {
        const res = await fetch(`/api/gloss_files?category=${category}&folder=${encodeURIComponent(folder)}`);
        const data = await res.json();

        if (!data.frames || data.frames.length === 0) {
            if (activeFileEl) activeFileEl.innerHTML = `${folder} <span style="font-size:11px; font-weight:normal; padding:2px 6px; border-radius:3px; background:#f44336; color:white; margin-left:8px; vertical-align:middle;">No NPZ Files</span>`;
            return;
        }

        const fetchResults = await Promise.all(
            data.frames.map(async (frame) => {
                const npzRes = await fetch(frame.npz_url);
                if (!npzRes.ok) return null;
                const buffer = await npzRes.arrayBuffer();
                return {
                    file: new File([buffer], frame.filename),
                    index: frame.index,
                    img_url: frame.img_url
                };
            })
        );

        let simulatedFiles = [];
        let imagesDict = {};
        
	    for (const r of fetchResults) {
            if (!r) continue;
            simulatedFiles.push(r.file);
            imagesDict[r.index] = r.img_url;
        }
	prefetchImages(imagesDict);
        if (simulatedFiles.length > 0) {
            if (activeFileEl) {
                activeFileEl.innerHTML = `${folder} <span style="font-size:11px; font-weight:normal; padding:2px 6px; border-radius:3px; background:#4CAF50; color:white; margin-left:8px; vertical-align:middle;">SMPL-X NPZ</span>`;
            }
            processNPZFiles(simulatedFiles, imagesDict, cacheKey);
        }

    } catch (e) {
        console.error(`Error loading gloss ${folder}:`, e);
        if (activeFileEl) {
            activeFileEl.innerHTML = `${folder} <span style="font-size:11px; font-weight:normal; padding:2px 6px; border-radius:3px; background:#f44336; color:white; margin-left:8px; vertical-align:middle;">Error Loading</span>`;
        }
    }
}
// ============================================================================
// CUSTOM NPZ UPLOAD (DRAG & DROP)
// ============================================================================

const handsMeanLeft = [0.11167871206998825, 0.042892176657915115, -0.41644182801246643, 0.10881132632493973, -0.06598567962646484, -0.7562199831008911, -0.09639296680688858, -0.09091565757989883, -0.18845929205417633, -0.1180950403213501, 0.050943851470947266, -0.529584527015686, -0.14369840919971466, 0.055241700261831284, -0.7048571109771729, -0.01918291673064232, -0.09233684837818146, -0.33791351318359375, -0.4570329785346985, -0.1962839514017105, -0.6254575252532959, -0.21465237438678741, -0.06599828600883484, -0.5068942308425903, -0.3697243630886078, -0.060344625264406204, -0.07949022948741913, -0.1418696939945221, -0.08585263043642044, -0.6355282664299011, -0.3033415973186493, -0.05788097530603409, -0.6313892006874084, -0.17612089216709137, -0.13209307193756104, -0.37335458397865295, 0.8509643077850342, 0.27692273259162903, -0.09154807031154633, -0.4998394250869751, 0.02655647136271, 0.05288087576627731, 0.5355591773986816, 0.04596104100346565, -0.2773580253124237];
const handsMeanRight = [0.11167871206998825, -0.042892176657915115, 0.41644182801246643, 0.10881132632493973, 0.06598567962646484, 0.7562199831008911, -0.09639296680688858, 0.09091565757989883, 0.18845929205417633, -0.1180950403213501, -0.050943851470947266, 0.529584527015686, -0.14369840919971466, -0.055241700261831284, 0.7048571109771729, -0.01918291673064232, 0.09233684837818146, 0.33791351318359375, -0.4570329785346985, 0.1962839514017105, 0.6254575252532959, -0.21465237438678741, 0.06599828600883484, 0.5068942308425903, -0.3697243630886078, 0.060344625264406204, 0.07949022948741913, -0.1418696939945221, 0.08585263043642044, 0.6355282664299011, -0.3033415973186493, 0.05788097530603409, 0.6313892006874084, -0.17612089216709137, 0.13209307193756104, 0.37335458397865295, 0.8509643077850342, -0.27692273259162903, 0.09154807031154633, -0.4998394250869751, -0.02655647136271, -0.05288087576627731, 0.5355591773986816, -0.04596104100346565, 0.2773580253124237];

function parseNPY(uint8arr) {
    // uint8arr must be a Uint8Array (NOT using .buffer which loses byteOffset in fflate)
    // NPY format: \x93NUMPY + version(2 bytes) + header_len + header_string + data
    const major = uint8arr[6];
    let headerLen, headerStart;
    if (major >= 2) {
        // Version 2+: 4-byte header length
        headerLen = uint8arr[8] | (uint8arr[9] << 8) | (uint8arr[10] << 16) | (uint8arr[11] << 24);
        headerStart = 12;
    } else {
        // Version 1: 2-byte header length
        headerLen = uint8arr[8] | (uint8arr[9] << 8);
        headerStart = 10;
    }
    const dataOffset = headerStart + headerLen;
    
    // Parse header to get dtype
    const headerStr = new TextDecoder().decode(uint8arr.slice(headerStart, dataOffset));
    const isFloat64 = headerStr.includes('f8') || headerStr.includes('float64');
    
    // Extract raw data bytes into a NEW aligned ArrayBuffer
    const dataBytes = uint8arr.slice(dataOffset);
    const alignedBuf = dataBytes.buffer.slice(dataBytes.byteOffset, dataBytes.byteOffset + dataBytes.byteLength);
    
    if (isFloat64) {
        // Read as float64 then convert to float32
        const f64 = new Float64Array(alignedBuf);
        return new Float32Array(f64); // JS auto-converts
    } else {
        return new Float32Array(alignedBuf);
    }
}

function processNPZFiles(files, imagesDict = {},cacheKey = null) {
    // Sort files by frame number in name (e.g., 0000_smplx.npz)
    files.sort((a, b) => {
        let matchA = a.name.match(/\d+/);
        let matchB = b.name.match(/\d+/);
        let na = matchA ? parseInt(matchA[0], 10) : 0;
        let nb = matchB ? parseInt(matchB[0], 10) : 0;
        return na - nb;
    });
    
    let loadedCount = 0;
    let newFrames = new Array(files.length);
    const flipQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    
    files.forEach((file, idx) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const arrayBuffer = e.target.result;
                const unzipped = fflate.unzipSync(new Uint8Array(arrayBuffer));
                
                // CRITICAL: fflate returns Uint8Arrays that may share an underlying buffer.
                // Pass the Uint8Array directly to parseNPY, NOT .buffer (which loses byteOffset).
                const getArr = (name, size) => {
                    const entry = unzipped[name + '.npy'];
                    if (!entry) return new Float32Array(size);
                    return parseNPY(entry);
                };
                
                const go = getArr('global_orient', 3);
                const bp = getArr('body_pose', 63);
                const jp = getArr('jaw_pose', 3);
                const lep = getArr('leye_pose', 3);
                const rep = getArr('reye_pose', 3);
                const lhp = getArr('left_hand_pose', 45);
                const rhp = getArr('right_hand_pose', 45);
                const tr = getArr('transl', 3);
                
                // ALWAYS add hand mean: the SMPL-X forward model internally does
                // full_hand_pose = hand_pose_input + hand_mean
                // SMPLer-X saves the INPUT (before mean is added), so we must add it.
                // Verified from SMPLer-X/main/SMPLer_X.py line 81-83 and inference.py line 145.
                for (let i = 0; i < 45; i++) {
                    lhp[i] += handsMeanLeft[i];
                    rhp[i] += handsMeanRight[i];
                }
                
                // Construct standard 55-joint array
                let all_axis_angles = [];
                const addToList = (arr) => {
                    for(let i=0; i<arr.length; i+=3) {
                        all_axis_angles.push([arr[i], arr[i+1], arr[i+2]]);
                    }
                };
                
                addToList(go);
                addToList(bp);
                addToList(jp);
                addToList(lep);
                addToList(rep);
                addToList(lhp);
                addToList(rhp);
                
                let frameRotations = [];
                for (let i = 0; i < 55; i++) {
                    let vec = new THREE.Vector3(all_axis_angles[i][0], all_axis_angles[i][1], all_axis_angles[i][2]);
                    let angle = vec.length();
                    let q = new THREE.Quaternion();
                    if (angle > 1e-6) {
                        vec.normalize();
                        q.setFromAxisAngle(vec, angle);
                    }
                    
                    // Apply camera coordinate flip to global_orient (pelvis)
                    if (i === 0) {
                        q.premultiply(flipQ);
                    }

                    // Perfectly straighten the pelvis, spine, and legs
                    // to remove incorrect mocap noise (indices 0 through 11).
                    // This must run AFTER the flip logic above so the pelvis remains perfectly neutral (Identity).
                    if (activeCategory === 'raw' && i >= 0 && i <= 11) {
                        q.identity();
                    }
                    
                    frameRotations.push([q.x, q.y, q.z, q.w]);
                }
                
                // Flip translation
                let tVec = new THREE.Vector3(tr[0], tr[1], tr[2]);
                tVec.applyQuaternion(flipQ);
                
                // Reset the translation to place avatar directly on the platform
                if (activeCategory === 'raw') {
                    tVec.set(0, 0, 0);
                }
                
                let match = file.name.match(/\d+/);
                const frameIndex = match ? parseInt(match[0], 10) : idx;
                newFrames[idx] = {
                    index: frameIndex,
                    rotations: frameRotations,
                    transl: [tVec.x, tVec.y, tVec.z],
                    imageUrl: imagesDict[frameIndex] || null
                };
                
                loadedCount++;
                if (loadedCount === files.length) {
                    // All files processed
                    animationFrames = newFrames.filter(Boolean);
                    // Deep clone original frames to support "Revert to Original NPZ"
                    originalAnimationFrames = JSON.parse(JSON.stringify(animationFrames));
                    
                    isDataLoaded = true;
                     if (cacheKey) {
                        glossDataCache[cacheKey] = {
                            animationFrames: JSON.parse(JSON.stringify(animationFrames)),
                            originalAnimationFrames: JSON.parse(JSON.stringify(originalAnimationFrames))
                        };
                    }
                    const slider = document.getElementById('frame-slider');
                    const totalDisplay = document.getElementById('total-frames-display');
                    if (slider) {
                        slider.max = animationFrames.length - 1;
                        slider.value = 0;
                    }
                    if (totalDisplay) totalDisplay.textContent = `(Keyframe 1 of ${animationFrames.length})`;
                    
                    goToFrame(0);
                    
                    // Auto-focus camera on upper body
                    setTimeout(() => {
                        const keyBoneNames = ['spine3', 'head', 'left_wrist', 'right_wrist'];
                        const positions = [];
                        keyBoneNames.forEach(name => {
                            if (boneObjects[name]) {
                                const p = new THREE.Vector3();
                                boneObjects[name].getWorldPosition(p);
                                positions.push(p);
                            }
                        });
                        
                        if (positions.length > 0) {
                            const box = new THREE.Box3();
                            positions.forEach(p => box.expandByPoint(p));
                            const center = new THREE.Vector3();
                            box.getCenter(center);
                            const size = new THREE.Vector3();
                            box.getSize(size);
                            
                            const maxDim = Math.max(size.x, size.y, size.z);
                            const dist = maxDim * 2.0; 
                            
                            controls.target.copy(center);
                            camera.position.set(center.x, center.y, center.z + dist);
                            controls.update();
                            
                            // Force resize to fix layout
                            window.dispatchEvent(new Event('resize'));
                        }
                    }, 150);
                }
            } catch (err) {
                console.error("Error processing NPZ:", err);
            }
        };
        reader.readAsArrayBuffer(file);
    });
}

// Store last uploaded files for re-processing when toggle changes
// Store last uploaded files for re-processing when toggle changes
let lastUploadedFiles = null;

// ============================================================================
// UI EVENT BINDINGS
// ============================================================================
// Modal handlers
    const discardBtn = document.getElementById('modal-discard');
    if (discardBtn) {
        discardBtn.addEventListener('click', () => {
            document.getElementById('unsaved-modal').style.display = 'none';
            // Restore from backup
            if (frameBackupRotations && animationFrames[currentFrame]) {
                animationFrames[currentFrame].rotations = JSON.parse(JSON.stringify(frameBackupRotations));
                applyFrameData(animationFrames[currentFrame]);
                updateUIForSelectedBone();
            }
            setUnsavedChanges(false);
            goToFrame(pendingFrameChange);
        });
    }

    const modalSaveBtn = document.getElementById('modal-save');
    if (modalSaveBtn) {
        modalSaveBtn.addEventListener('click', () => {
            document.getElementById('unsaved-modal').style.display = 'none';
            if (selectedBoneName) persistBoneChange(selectedBoneName);
            setUnsavedChanges(false);
            goToFrame(pendingFrameChange);
        });
    }

const showAllBonesToggle = document.getElementById('show-all-bones-toggle');
if (showAllBonesToggle) {
    showAllBonesToggle.addEventListener('change', updateBoneVisibility);
}

const enableGizmoToggle = document.getElementById('enable-gizmo-toggle');
if (enableGizmoToggle) {
    enableGizmoToggle.addEventListener('change', () => {
        if (selectedBoneName) showGizmoAtBone(selectedBoneName);
    });
}

function updateBoneVisibility() {
    const showAll = showAllBonesToggle ? showAllBonesToggle.checked : true;
    boneHitMeshes.forEach(mesh => {
        if (mesh.userData.visibleBone) {
            const isHovered = (hoveredBoneHit === mesh);
            const isSelected = (selectedBoneName === mesh.userData.boneName);
            if (isSelected) {
                mesh.userData.visibleBone.material.color.setHex(0xff4444);
                mesh.userData.visibleBone.material.opacity = 0.8;
            } else if (isHovered) {
                mesh.userData.visibleBone.material.color.setHex(0x00ff88);
                mesh.userData.visibleBone.material.opacity = 0.6;
            } else {
                mesh.userData.visibleBone.material.color.setHex(0xffffff);
                mesh.userData.visibleBone.material.opacity = showAll ? 0.6 : 0.0;
            }
        }
    });
}

// Ensure hovering/selecting works with the visibility toggle
const originalSelectBone = selectBone;
selectBone = function(boneName) {
    originalSelectBone(boneName);
    updateBoneVisibility();
};

const originalDeselectBone = deselectBone;
deselectBone = function() {
    originalDeselectBone();
    updateBoneVisibility();
};

const saveAllBtn = document.getElementById('save-all-btn');
if (saveAllBtn) {
    saveAllBtn.addEventListener('click', async () => {
        if (!isDataLoaded || animationFrames.length === 0) {
            alert('No data loaded to save!');
            return;
        }
        
        if (!activeGlossFolder) {
            alert('No gloss loaded!');
            return;
        }

        saveAllBtn.textContent = 'Saving...';
        saveAllBtn.disabled = true;

        try {
            const promises = [];
            
            // Loop through every frame, create an NPZ structure for it
            for (let i = 0; i < animationFrames.length; i++) {
                const frame = animationFrames[i];
                
                // Convert back from Quaternions to Axis-Angles for 55 bones
                const go = new Float32Array(3);
                const bp = new Float32Array(63); // 21 * 3
                const jp = new Float32Array(3);
                const lep = new Float64Array(3); // eyes are f8
                const rep = new Float64Array(3); // eyes are f8
                const lhp = new Float32Array(45); // 15 * 3
                const rhp = new Float32Array(45); // 15 * 3
                const tr = new Float32Array(frame.transl);
                
                const flipQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
                const flipQInv = flipQ.clone().invert();
                
                let outIdx = 0;
                for (let b = 0; b < 55; b++) {
                    const q = new THREE.Quaternion(frame.rotations[b][0], frame.rotations[b][1], frame.rotations[b][2], frame.rotations[b][3]);
                    if (b === 0) {
                        q.premultiply(flipQInv); // undo camera flip
                    }
                    
                    const angle = 2 * Math.acos(Math.max(-1, Math.min(1, q.w)));
                    const sin = Math.sqrt(1 - q.w * q.w);
                    let x = 0, y = 0, z = 0;
                    if (sin > 0.0001) {
                        x = q.x / sin * angle;
                        y = q.y / sin * angle;
                        z = q.z / sin * angle;
                    }
                    
                    if (b === 0) {
                        go[0] = x; go[1] = y; go[2] = z;
                    } else if (b >= 1 && b <= 21) {
                        bp[(b-1)*3] = x; bp[(b-1)*3+1] = y; bp[(b-1)*3+2] = z;
                    } else if (b === 22) {
                        jp[0] = x; jp[1] = y; jp[2] = z;
                    } else if (b === 23) {
                        lep[0] = x; lep[1] = y; lep[2] = z;
                    } else if (b === 24) {
                        rep[0] = x; rep[1] = y; rep[2] = z;
                    } else if (b >= 25 && b <= 39) {
                        let hb = b - 25;
                        lhp[hb*3] = x; lhp[hb*3+1] = y; lhp[hb*3+2] = z;
                    } else if (b >= 40 && b <= 54) {
                        let hb = b - 40;
                        rhp[hb*3] = x; rhp[hb*3+1] = y; rhp[hb*3+2] = z;
                    }
                }
                
                // Subtract hand mean
                for (let k = 0; k < 45; k++) {
                    lhp[k] -= handsMeanLeft[k];
                    rhp[k] -= handsMeanRight[k];
                }
                
                // Un-flip translation
                let tVec = new THREE.Vector3(tr[0], tr[1], tr[2]);
                tVec.applyQuaternion(flipQInv);
                tr[0] = tVec.x; tr[1] = tVec.y; tr[2] = tVec.z;

                // Function to build an NPY file (version 1)
                const buildNPY = (arr, dtype, shapeStr) => {
                    const dictStr = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
                    const paddingLen = 16 - ((10 + dictStr.length + 1) % 16);
                    const headerStr = dictStr + ' '.repeat(paddingLen) + '\n';
                    const headerLen = headerStr.length;
                    
                    const out = new Uint8Array(10 + headerLen + arr.byteLength);
                    out.set([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0], 0);
                    out[8] = headerLen & 0xff;
                    out[9] = (headerLen >> 8) & 0xff;
                    for (let c = 0; c < headerLen; c++) {
                        out[10+c] = headerStr.charCodeAt(c);
                    }
                    out.set(new Uint8Array(arr.buffer), 10 + headerLen);
                    return out;
                };

                const npzContent = {};
                npzContent['global_orient.npy'] = buildNPY(go, '<f4', '(1, 3)');
                npzContent['body_pose.npy'] = buildNPY(bp, '<f4', '(21, 3)');
                npzContent['jaw_pose.npy'] = buildNPY(jp, '<f4', '(1, 3)');
                npzContent['leye_pose.npy'] = buildNPY(lep, '<f8', '(1, 3)');
                npzContent['reye_pose.npy'] = buildNPY(rep, '<f8', '(1, 3)');
                npzContent['left_hand_pose.npy'] = buildNPY(lhp, '<f4', '(15, 3)');
                npzContent['right_hand_pose.npy'] = buildNPY(rhp, '<f4', '(15, 3)');
                npzContent['transl.npy'] = buildNPY(tr, '<f4', '(1, 3)');
                
                const npzBytes = fflate.zipSync(npzContent);
                const frameStr = String(frame.index).padStart(5, '0');
                const filename = `${frameStr}_0.npz`;
                
                const uploadPromise = fetch(`/api/save_file?folder=${encodeURIComponent(activeGlossFolder)}&filename=${encodeURIComponent(filename)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/octet-stream' },
                    body: npzBytes
                }).then(r => r.json());
                
                promises.push(uploadPromise);
            }
            
            await Promise.all(promises);
            
            alert(`Successfully saved to ${activeGlossFolder.endsWith('_corrected') ? activeGlossFolder : activeGlossFolder + '_corrected'}!`);
            setUnsavedChanges(false);
        } catch (e) {
            console.error(e);
            alert(`Failed to save: ${e.message}`);
        } finally {
            saveAllBtn.textContent = 'Save All Changes';
            saveAllBtn.disabled = false;
        }
    });
}
const saveBtn = document.getElementById('save-btn');
if (saveBtn) {
    saveBtn.addEventListener('click', () => {
        if (!selectedBoneName) return;
        persistBoneChange(selectedBoneName);
        setUnsavedChanges(false);
        alert('Current frame edits saved to memory!');
    });
}

const discardBtnSidebar = document.getElementById('discard-btn');
if (discardBtnSidebar) {
    discardBtnSidebar.addEventListener('click', () => {
        if (!frameBackupRotations || !animationFrames[currentFrame]) return;
        // Restore rotations to backup
        animationFrames[currentFrame].rotations = JSON.parse(JSON.stringify(frameBackupRotations));
        applyFrameData(animationFrames[currentFrame]);
        updateUIForSelectedBone();
        setUnsavedChanges(false);
    });
}

const revertOriginalBtn = document.getElementById('revert-original-btn');
if (revertOriginalBtn) {
    revertOriginalBtn.addEventListener('click', () => {
        if (!originalAnimationFrames[currentFrame] || !animationFrames[currentFrame]) return;
        if (!confirm("Are you sure you want to revert this frame exactly to how it was downloaded, erasing ALL edits?")) return;
        
        // Deep clone from original
        animationFrames[currentFrame].rotations = JSON.parse(JSON.stringify(originalAnimationFrames[currentFrame].rotations));
        
        // Backup is also reset so 'discard' is meaningless
        frameBackupRotations = JSON.parse(JSON.stringify(originalAnimationFrames[currentFrame].rotations));
        
        applyFrameData(animationFrames[currentFrame]);
        updateUIForSelectedBone();
        setUnsavedChanges(false);
    });
}

// Left/Right hand index mapping
const LEFT_ARM_INDICES = [13, 16, 18, 20]; // collar, shoulder, elbow, wrist
for(let i=25; i<=39; i++) LEFT_ARM_INDICES.push(i); // fingers

const RIGHT_ARM_INDICES = [14, 17, 19, 21]; // collar, shoulder, elbow, wrist
for(let i=40; i<=54; i++) RIGHT_ARM_INDICES.push(i); // fingers

async function fetchDefaultArmPose(side) {
    if (!animationFrames[currentFrame]) return;
    
    try {
        const res = await fetch(`/sign_annotation/default/pose.npz`);
        if (!res.ok) throw new Error(`Could not load pose.npz`);
        
        const buffer = await res.arrayBuffer();
        const unzipped = fflate.unzipSync(new Uint8Array(buffer));
        
        const getArr = (name, size) => {
            const entry = unzipped[name + '.npy'];
            if (!entry) return new Float32Array(size);
            return parseNPY(entry);
        };
        
        const go = getArr('global_orient', 3);
        const bp = getArr('body_pose', 63);
        const jp = getArr('jaw_pose', 3);
        const lep = getArr('leye_pose', 3);
        const rep = getArr('reye_pose', 3);
        const lhp = getArr('left_hand_pose', 45);
        const rhp = getArr('right_hand_pose', 45);
        
        // Apply mean hands exactly like in processNPZFiles
        for (let i = 0; i < 45; i++) {
            lhp[i] += handsMeanLeft[i];
            rhp[i] += handsMeanRight[i];
        }
        
        let all_axis_angles = [];
        const addToList = (arr) => {
            for(let i=0; i<arr.length; i+=3) {
                all_axis_angles.push([arr[i], arr[i+1], arr[i+2]]);
            }
        };
        addToList(go); addToList(bp); addToList(jp); addToList(lep); addToList(rep); addToList(lhp); addToList(rhp);
        
        const targetIndices = side === 'left' ? LEFT_ARM_INDICES : RIGHT_ARM_INDICES;
        
        for (let idx of targetIndices) {
            let vec = new THREE.Vector3(all_axis_angles[idx][0], all_axis_angles[idx][1], all_axis_angles[idx][2]);
            let angle = vec.length();
            let q = new THREE.Quaternion();
            if (angle > 1e-6) {
                vec.normalize();
                q.setFromAxisAngle(vec, angle);
            }
            animationFrames[currentFrame].rotations[idx] = [q.x, q.y, q.z, q.w];
        }
        
        applyFrameData(animationFrames[currentFrame]);
        updateUIForSelectedBone();
        setUnsavedChanges(true);
        
        alert(`Successfully injected default ${side} hand pose!`);
        
    } catch (e) {
        console.error(e);
        alert(`Error loading default ${side} hand: ` + e.message);
    }
}

const loadDefaultLeftBtn = document.getElementById('load-default-left-btn');
if (loadDefaultLeftBtn) {
    loadDefaultLeftBtn.addEventListener('click', () => fetchDefaultArmPose('left'));
}

const loadDefaultRightBtn = document.getElementById('load-default-right-btn');
if (loadDefaultRightBtn) {
    loadDefaultRightBtn.addEventListener('click', () => fetchDefaultArmPose('right'));
}

function handleManualRotationInput() {
    if (!selectedBoneName) return;
    const x = parseFloat(document.getElementById('rot-x-val').value || 0) * Math.PI / 180;
    const y = parseFloat(document.getElementById('rot-y-val').value || 0) * Math.PI / 180;
    const z = parseFloat(document.getElementById('rot-z-val').value || 0) * Math.PI / 180;
    
    activeEuler.set(x, y, z, 'XYZ');
    
    const bone = boneObjects[selectedBoneName];
    if (bone) {
        bone.quaternion.setFromEuler(activeEuler);
        bone.updateMatrixWorld(true);
        persistBoneChange(selectedBoneName);
        showGizmoAtBone(selectedBoneName);
    }
}

['x', 'y', 'z'].forEach(axis => {
    const input = document.getElementById(`rot-${axis}-val`);
    if (input) {
        input.addEventListener('change', handleManualRotationInput);
        input.addEventListener('input', handleManualRotationInput); // Fire instantly on native up/down arrows
        input.addEventListener('wheel', (e) => {
            e.preventDefault();
            const step = e.shiftKey ? 10 : 1;
            const dir = e.deltaY > 0 ? -1 : 1;
            input.value = (parseFloat(input.value || 0) + step * dir).toFixed(1);
            handleManualRotationInput();
        }, { passive: false });
    }
});

function handleManualTranslationInput() {
    if (!animationFrames || animationFrames.length === 0) return;
    const x = parseFloat(document.getElementById('trans-x-val').value || 0);
    const y = parseFloat(document.getElementById('trans-y-val').value || 0);
    const z = parseFloat(document.getElementById('trans-z-val').value || 0);
    
    const frameData = animationFrames[currentFrame];
    if (!frameData.transl) frameData.transl = [0, 0, 0];
    
    frameData.transl[0] = x;
    frameData.transl[1] = y;
    frameData.transl[2] = z;
    
    setUnsavedChanges(true);
    applyFrameData(frameData);
}

['x', 'y', 'z'].forEach(axis => {
    const input = document.getElementById(`trans-${axis}-val`);
    if (input) {
        input.addEventListener('change', handleManualTranslationInput);
        input.addEventListener('input', handleManualTranslationInput);
        input.addEventListener('wheel', (e) => {
            e.preventDefault();
            const step = e.shiftKey ? 0.1 : 0.01;
            const dir = e.deltaY > 0 ? -1 : 1;
            input.value = (parseFloat(input.value || 0) + step * dir).toFixed(3);
            handleManualTranslationInput();
        }, { passive: false });
    }
});

function attachAdjustmentListener(id, axis, type, sign) {
        const btn = document.getElementById(id);
        if (!btn) return;
        btn.addEventListener('click', () => {
            if (type === 'rot') {
                if (!selectedBoneName) return;
                const bone = boneObjects[selectedBoneName];
                if (!bone) return;
                
                setUnsavedChanges(true);
                const step = 5 * Math.PI / 180;
                if (lastSelectedBoneForEuler !== selectedBoneName) {
                    activeEuler.setFromQuaternion(bone.quaternion, 'XYZ');
                    lastSelectedBoneForEuler = selectedBoneName;
                }
                activeEuler[axis] += sign * step;
                bone.quaternion.setFromEuler(activeEuler);
                bone.updateMatrixWorld(true);
                persistBoneChange(selectedBoneName);
                updateUIForSelectedBone(true);
            } else if (type === 'trans') {
                setUnsavedChanges(true);
                const step = 0.05;
                const frameData = animationFrames[currentFrame];
                if (!frameData.transl) frameData.transl = [0,0,0];
                const axisIdx = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
                frameData.transl[axisIdx] += sign * step;
                applyFrameData(frameData);
                updateTranslationUI();
            }
        });
    }

    ['x','y','z'].forEach(axis => {
        attachAdjustmentListener(`rot-${axis}-minus`, axis, 'rot', -1);
        attachAdjustmentListener(`rot-${axis}-plus`,  axis, 'rot', 1);
        
        attachAdjustmentListener(`trans-${axis}-minus`, axis, 'trans', -1);
        attachAdjustmentListener(`trans-${axis}-plus`,  axis, 'trans', 1);
        
        const row = document.getElementById(`rot-${axis}-row`);
        if (row) {
            row.addEventListener('wheel', (e) => {
                if (!selectedBoneName) return;
                e.preventDefault();
                const sign = e.deltaY > 0 ? -1 : 1;
                const btn = document.getElementById(`rot-${axis}-${sign > 0 ? 'plus' : 'minus'}`);
                if (btn) btn.click();
            }, { passive: false });
        }
        
        const transRow = document.getElementById(`trans-${axis}-row`);
        if (transRow) {
            transRow.addEventListener('wheel', (e) => {
                e.preventDefault();
                const sign = e.deltaY > 0 ? -1 : 1;
                const btn = document.getElementById(`trans-${axis}-${sign > 0 ? 'plus' : 'minus'}`);
                if (btn) btn.click();
            }, { passive: false });
        }
    });

    document.getElementById('prev-frame-btn')?.addEventListener('click', () => {
        if (currentFrame > 0) {
            if (hasUnsavedChanges) showUnsavedModal(currentFrame - 1);
            else goToFrame(currentFrame - 1);
        }
    });

    document.getElementById('next-frame-btn')?.addEventListener('click', () => {
        if (currentFrame < animationFrames.length - 1) {
            if (hasUnsavedChanges) showUnsavedModal(currentFrame + 1);
            else goToFrame(currentFrame + 1);
	}
   
    });

    document.getElementById('copy-specific-frame-btn')?.addEventListener('click', () => {
        const input = document.getElementById('copy-frame-input');
        if (!input || input.value === '') {
            alert("Please enter a valid frame number.");
            return;
        }
        const inputValue = parseInt(input.value, 10);
        if (isNaN(inputValue)) {
            alert("Invalid frame number.");
            return;
        }

        console.log(`[DEBUG] Attempting to copy pose from keyframe number: ${inputValue}`);
        console.log(`[DEBUG] Current internal frame index (currentFrame): ${currentFrame}`);

        let targetFrameIndex = inputValue - 1;

        if (targetFrameIndex < 0 || targetFrameIndex >= animationFrames.length) {
            console.error(`[DEBUG] Keyframe ${inputValue} is out of bounds.`);
            alert(`Keyframe ${inputValue} is out of bounds. Please enter a number between 1 and ${animationFrames.length}.`);
            return;
        }

        console.log(`[DEBUG] Using internal array index ${targetFrameIndex} for keyframe ${inputValue}.`);

        if (targetFrameIndex === currentFrame) {
            console.warn(`[DEBUG] Target frame and current frame are the same (${currentFrame}). Nothing to change.`);
            alert(`You are already on keyframe ${inputValue}!`);
            return;
        }

        const sourceRotations = animationFrames[targetFrameIndex].rotations;
        const currentRotations = animationFrames[currentFrame].rotations;
        
        console.log(`[DEBUG] sourceRotations length:`, sourceRotations?.length);
        console.log(`[DEBUG] currentRotations length before copy:`, currentRotations?.length);
        console.log(`[DEBUG] sourceRotations[0] (pelvis):`, sourceRotations?.[0]);
        console.log(`[DEBUG] currentRotations[0] (pelvis) BEFORE copy:`, currentRotations?.[0]);

        if (sourceRotations && currentRotations) {
            for(let i=0; i<sourceRotations.length; i++) {
                currentRotations[i] = [...sourceRotations[i]];
            }
            console.log(`[DEBUG] currentRotations[0] (pelvis) AFTER copy:`, currentRotations[0]);
            
            setUnsavedChanges(true);
            applyFrameData(animationFrames[currentFrame]);
            updateUIForSelectedBone();
            
            console.log(`[DEBUG] Successfully applied frame data to skeleton!`);
            alert(`Copied pose from frame ${inputValue} successfully!`);
        } else {
            console.error(`[DEBUG] Missing rotations data! source:`, !!sourceRotations, `current:`, !!currentRotations);
        }
    });

    const copyLFingersBtn = document.getElementById('copy-l-fingers-btn');
    if (copyLFingersBtn) {
        copyLFingersBtn.addEventListener('click', () => {
            const input = document.getElementById('copy-frame-input');
            const inputValue = parseInt(input.value, 10);
            if (isNaN(inputValue)) return alert("Invalid frame number.");
            
            let targetFrameIndex = inputValue - 1;
            if (targetFrameIndex < 0 || targetFrameIndex >= animationFrames.length) {
                return alert(`Keyframe ${inputValue} is out of bounds.`);
            }
            if (targetFrameIndex === currentFrame) return alert(`You are already on keyframe ${inputValue}!`);
            
            const sourceRotations = animationFrames[targetFrameIndex].rotations;
            const currentRotations = animationFrames[currentFrame].rotations;
            
            if (sourceRotations && currentRotations) {
                // Left fingers are indices 25 to 39
                for (let i = 25; i <= 39; i++) {
                    currentRotations[i] = [...sourceRotations[i]];
                }
                setUnsavedChanges(true);
                applyFrameData(animationFrames[currentFrame]);
                updateUIForSelectedBone();
                alert(`Transferred LEFT fingers from frame ${inputValue}!`);
            }
        });
    }

    const copyRFingersBtn = document.getElementById('copy-r-fingers-btn');
    if (copyRFingersBtn) {
        copyRFingersBtn.addEventListener('click', () => {
            const input = document.getElementById('copy-frame-input');
            const inputValue = parseInt(input.value, 10);
            if (isNaN(inputValue)) return alert("Invalid frame number.");
            
            let targetFrameIndex = inputValue - 1;
            if (targetFrameIndex < 0 || targetFrameIndex >= animationFrames.length) {
                return alert(`Keyframe ${inputValue} is out of bounds.`);
            }
            if (targetFrameIndex === currentFrame) return alert(`You are already on keyframe ${inputValue}!`);
            
            const sourceRotations = animationFrames[targetFrameIndex].rotations;
            const currentRotations = animationFrames[currentFrame].rotations;
            
            if (sourceRotations && currentRotations) {
                // Right fingers are indices 40 to 54
                for (let i = 40; i <= 54; i++) {
                    currentRotations[i] = [...sourceRotations[i]];
                }
                setUnsavedChanges(true);
                applyFrameData(animationFrames[currentFrame]);
                updateUIForSelectedBone();
                alert(`Transferred RIGHT fingers from frame ${inputValue}!`);
            }
        });
    }

function focusCameraOn(targetPos, distance) {
    controls.target.copy(targetPos);
    const dir = camera.position.clone().sub(targetPos).normalize();
    camera.position.copy(targetPos).add(dir.multiplyScalar(distance));
    controls.update();
}

document.getElementById('focus-full')?.addEventListener('click', () => {
    const rootBoneName = SMPLX_BONE_NAMES[0];
    if (boneObjects[rootBoneName]) {
        const rootPos = new THREE.Vector3();
        boneObjects[rootBoneName].getWorldPosition(rootPos);
        rootPos.y += 0.5;
        focusCameraOn(rootPos, 3.0);
    } else {
        focusCameraOn(new THREE.Vector3(0, 1, 0), 3.0); 
    }
});

document.getElementById('focus-left')?.addEventListener('click', () => {
    const lHandName = SMPLX_BONE_NAMES[20];
    if (boneObjects[lHandName]) {
        const pos = new THREE.Vector3();
        boneObjects[lHandName].getWorldPosition(pos);
        focusCameraOn(pos, 1.0);
    }
});

document.getElementById('focus-right')?.addEventListener('click', () => {
    const rHandName = SMPLX_BONE_NAMES[21];
    if (boneObjects[rHandName]) {
        const pos = new THREE.Vector3();
        boneObjects[rHandName].getWorldPosition(pos);
        focusCameraOn(pos, 1.0);
    }
});

// ============================================================================
// GIZMO HELPERS
// ============================================================================
function showGizmoAtBone(boneName) {
    if (!rotationGizmo || !boneName) return;
    
    const enableGizmo = document.getElementById('enable-gizmo-toggle')?.checked;
    if (!enableGizmo) {
        rotationGizmo.visible = false;
        return;
    }
    
    const bone = boneObjects[boneName];
    if (!bone) return;
    
    const worldPos = new THREE.Vector3();
    bone.getWorldPosition(worldPos);
    rotationGizmo.position.copy(worldPos);
    rotationGizmo.visible = true;
}

function hideGizmo() {
    if (rotationGizmo) rotationGizmo.visible = false;
}

function selectBone(boneName) {
    selectedBoneName = boneName;
    updateBoneVisibility();
    showGizmoAtBone(boneName);
    updateUIForSelectedBone();
}

function deselectBone() {
    selectedBoneName = null;
    hideGizmo();
    updateBoneVisibility();
    updateUIForSelectedBone();
}

document.getElementById('deselect-bone-btn')?.addEventListener('click', () => {
    deselectBone();
});

function persistBoneChange(boneName) {
    const bone = boneObjects[boneName];
    if (!bone || !animationFrames[currentFrame]) return;
    
    setUnsavedChanges(true);
    targetQuaternions[boneName].copy(bone.quaternion);
    
    // Save back to JSON array in memory (always in SMPLX format)
    const boneIdx = SMPLX_BONE_NAMES.indexOf(boneName);
    if (boneIdx === -1) return;
    if (!animationFrames[currentFrame].rotations[boneIdx]) {
        animationFrames[currentFrame].rotations[boneIdx] = [0,0,0,1];
    }
    
    const config = AVATAR_CONFIGS[activeAvatarKey];
    if (config && config.needsRetarget && restQuaternions[boneName]) {
        const invRest = restQuaternions[boneName].clone().invert();
        let smplxQ = invRest.multiply(bone.quaternion.clone());
        if (config.unmapRotation) {
            smplxQ = config.unmapRotation(smplxQ, boneName);
        }
        animationFrames[currentFrame].rotations[boneIdx] = [smplxQ.x, smplxQ.y, smplxQ.z, smplxQ.w];
    } else {
        const q = bone.quaternion;
        animationFrames[currentFrame].rotations[boneIdx] = [q.x, q.y, q.z, q.w];
    }
}

// ============================================================================
// MOUSE EVENT HANDLERS
// ============================================================================
let hoveredBoneHit = null;

function getMouseNDC(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
}

function getRotAngle(e, center, axis) {
    getMouseNDC(e);
    const centerScreen = center.clone().project(camera);
    const dx = mouse.x - centerScreen.x;
    const dy = mouse.y - centerScreen.y;
    return Math.atan2(dy, dx);
}

window.addEventListener('mousemove', (e) => {
    getMouseNDC(e);
    
    if (isRotDragging && rotDragAxis && selectedBoneName) {
        const currentAngle = getRotAngle(e, rotDragBoneWorldPos, rotDragAxis);
        const deltaAngle = (currentAngle - rotDragStartAngle) * 0.2; 
        
        const bone = boneObjects[selectedBoneName];
        if (bone) {
            const axisVec = new THREE.Vector3(
                rotDragAxis === 'x' ? 1 : 0,
                rotDragAxis === 'y' ? 1 : 0,
                rotDragAxis === 'z' ? 1 : 0
            );
            const deltaQ = new THREE.Quaternion().setFromAxisAngle(axisVec, deltaAngle);
            bone.quaternion.copy(rotDragStartQuat).premultiply(deltaQ);
            
            bone.updateMatrixWorld(true);

            
            persistBoneChange(selectedBoneName);
            showGizmoAtBone(selectedBoneName);
            updateUIForSelectedBone();
        }
        return;
    }
    
    raycaster.setFromCamera(mouse, camera);
    
    if (rotationGizmo && rotationGizmo.visible) {
        const ringIntersects = raycaster.intersectObjects(ringMeshes, false);
        ringMeshes.forEach(r => {
            const baseColors = { x: 0xff4444, y: 0x44ff44, z: 0x4488ff };
            r.material.opacity = 0.85;
            r.material.color.setHex(baseColors[r.userData.axis]);
        });
        if (ringIntersects.length > 0) {
            const hitRing = ringIntersects[0].object;
            hitRing.material.opacity = 1.0;
            hitRing.material.color.setHex(0xffffff);
            document.body.style.cursor = 'grab';
            return;
        }
    }
    
    // If a bone is selected, lock selection (don't hover other bones)
    if (selectedBoneName) return;
    
    const boneIntersects = raycaster.intersectObjects(boneHitMeshes, false);
    if (boneIntersects.length > 0) {
        const hit = boneIntersects[0].object;
        if (hoveredBoneHit !== hit) {
            hoveredBoneHit = hit;
            updateBoneVisibility();
            if (hoveredBoneHit.userData.boneName !== selectedBoneName) {
                hoveredBoneHit.userData.visibleBone.material.color.setHex(0x00ff88);
                hoveredBoneHit.userData.visibleBone.material.opacity = 0.6;
            }
        }
        document.body.style.cursor = 'pointer';
    } else {
        if (hoveredBoneHit) {
            hoveredBoneHit = null;
            updateBoneVisibility();
        }
        document.body.style.cursor = 'default';
    }
});

window.addEventListener('beforeunload', (e) => {
    if (animationFrames && animationFrames.length > 0) {
        e.preventDefault();
        e.returnValue = '';
    }
});

window.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    getMouseNDC(e);
    raycaster.setFromCamera(mouse, camera);
    
    if (rotationGizmo && rotationGizmo.visible && selectedBoneName) {
        const ringIntersects = raycaster.intersectObjects(ringMeshes, false);
        if (ringIntersects.length > 0) {
            const hitRing = ringIntersects[0].object;
            isRotDragging = true;
            rotDragAxis = hitRing.userData.axis;
            
            const bone = boneObjects[selectedBoneName];
            bone.getWorldPosition(rotDragBoneWorldPos);
            rotDragStartQuat.copy(bone.quaternion);
            rotDragStartAngle = getRotAngle(e, rotDragBoneWorldPos, rotDragAxis);
            
            controls.enabled = false;
            document.body.style.cursor = 'grabbing';
            e.preventDefault();
            return;
        }
    }
    
    // If a bone is selected, prevent clicking anything else
    if (selectedBoneName) return;
    
    const boneIntersects = raycaster.intersectObjects(boneHitMeshes, false);
    if (boneIntersects.length > 0) {
        const hitMesh = boneIntersects[0].object;
        selectBone(hitMesh.userData.boneName);
        e.preventDefault();
        return;
    }
    deselectBone();
});

window.addEventListener('mouseup', (e) => {
    if (isRotDragging) {
        isRotDragging = false;
        rotDragAxis = null;
        controls.enabled = true;
        document.body.style.cursor = 'default';
        
        if (selectedBoneName) {
            persistBoneChange(selectedBoneName);
            updateUIForSelectedBone();
        }
    }
});

// ============================================================================
// RENDER LOOP
// ============================================================================
const clock = new THREE.Clock();
function animate() {
    requestAnimationFrame(animate);
    
    if (isPlaying && isDataLoaded && animationFrames.length > 0) {
        const delta = clock.getDelta();
        // Assuming ~30fps -> 0.033s per frame
        // For simplicity, change frame every 0.033s
        if (!window.frameTimer) window.frameTimer = 0;
        window.frameTimer += delta;
        
        if (window.frameTimer >= 0.165) {
            window.frameTimer = 0;
            currentFrame++;
            if (currentFrame >= animationFrames.length) {
                currentFrame = 0;
                isPlaying = false;
                const playBtn = document.getElementById('play-pause-btn');
                if (playBtn) playBtn.textContent = 'â–¶';
            }
            
            const slider = document.getElementById('frame-slider');
            if (slider) slider.value = currentFrame;
            
            applyFrameData(animationFrames[currentFrame]);
            
            if (selectedBoneName) {
                showGizmoAtBone(selectedBoneName);
            }
        }
    } else {
        clock.getDelta(); // keep it ticking
    }
    
    renderer.render(scene, camera);
}
animate();

// Sidebar toggle
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        sidebarToggle.textContent = sidebar.classList.contains('collapsed') ? 'â—€' : 'â–¶';
        
        // Dispatch resize smoothly during the 300ms CSS transition
        const start = Date.now();
        const interval = setInterval(() => {
            window.dispatchEvent(new Event('resize'));
            if (Date.now() - start > 350) clearInterval(interval);
        }, 16);
    });
}

// Resizer logic
const resizer = document.getElementById('resizer');
const leftPane = document.getElementById('image-pane');
const rightPane = document.getElementById('canvas-wrapper');
let isResizingPane = false;

if (resizer && leftPane && rightPane) {
    resizer.addEventListener('mousedown', (e) => {
        isResizingPane = true;
        document.body.style.cursor = 'col-resize';
        // Prevent iframes or other things from eating mouse events
        e.preventDefault();
    });

    window.addEventListener('mousemove', (e) => {
        if (!isResizingPane) return;
        
        const container = document.getElementById('avatar-container');
        if (!container) return;
        
        const containerRect = container.getBoundingClientRect();
        // Calculate percentage, capping between 10% and 90%
        let newFlexBasis = ((e.clientX - containerRect.left) / containerRect.width) * 100;
        newFlexBasis = Math.max(10, Math.min(90, newFlexBasis));
        
        leftPane.style.flex = `0 0 ${newFlexBasis}%`;
        
        // Let the right pane take up the rest of the space
        rightPane.style.flex = '1 1 0%';
    });

    window.addEventListener('mouseup', () => {
        if (isResizingPane) {
            isResizingPane = false;
            document.body.style.cursor = 'default';
            // Force a window resize event so ThreeJS canvas updates its aspect ratio
            window.dispatchEvent(new Event('resize'));
        }
    });
}

// Fullscreen API Logic
const fsImageBtn = document.getElementById('fs-image-btn');
const fsCanvasBtn = document.getElementById('fs-canvas-btn');

if (fsImageBtn) {
    fsImageBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            document.body.requestFullscreen().then(() => {
                document.body.classList.add('image-fullscreen-mode');
            }).catch(err => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    });
}

if (fsCanvasBtn) {
    fsCanvasBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
            document.body.requestFullscreen().then(() => {
                document.body.classList.add('model-fullscreen-mode');
            }).catch(err => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
        } else {
            document.exitFullscreen();
        }
    });
}

document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
        document.body.classList.remove('image-fullscreen-mode');
        document.body.classList.remove('model-fullscreen-mode');
        // Trigger resize to fix layout
        window.dispatchEvent(new Event('resize'));
    } else {
        window.dispatchEvent(new Event('resize'));
    }
});

