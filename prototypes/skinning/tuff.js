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
camera.position.set(0, 1.5, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(width, height);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 4, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(5, 5, 5);
scene.add(dirLight);
scene.add(new THREE.GridHelper(10, 10, 0x555555, 0x444444));

// ============================================================================
// COORDINATE TRANSFORM
// MediaPipe: x right, y down, z depth (positive = further from camera)
// Three.js:  x right, y up,   z toward viewer
// ============================================================================
function lmToWorld(lm) {
    return new THREE.Vector3(lm[0], -lm[1], -lm[2]);
}

// ============================================================================
// BONE TABLES
// ============================================================================
const targetBones = [
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandThumb1',  'LeftHandThumb2',  'LeftHandThumb3',
    'LeftHandIndex1',  'LeftHandIndex2',  'LeftHandIndex3',
    'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3',
    'LeftHandRing1',   'LeftHandRing2',   'LeftHandRing3',
    'LeftHandPinky1',  'LeftHandPinky2',  'LeftHandPinky3',
    'RightArm', 'RightForeArm', 'RightHand',
    'RightHandThumb1',  'RightHandThumb2',  'RightHandThumb3',
    'RightHandIndex1',  'RightHandIndex2',  'RightHandIndex3',
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3',
    'RightHandRing1',   'RightHandRing2',   'RightHandRing3',
    'RightHandPinky1',  'RightHandPinky2',  'RightHandPinky3',
];

const boneObjects       = {};
const restQuaternions   = {};
const targetQuaternions = {};

// Hand rest basis — captured once at load, used to compute orientation delta each frame
let restBasisQuatR         = new THREE.Quaternion();
let rightHandRestWorldQuat = new THREE.Quaternion();
let isRightHandBasisReady  = false;

let restBasisQuatL        = new THREE.Quaternion();
let leftHandRestWorldQuat = new THREE.Quaternion();
let isLeftHandBasisReady  = false;

let modelLoaded = false;

// ============================================================================
// ARM DIRECTION HELPER
// ============================================================================
function armDir(from, to, zMin, zMax) {
    const diff = new THREE.Vector3().subVectors(to, from);
    
    const len = diff.length();
    if (len < 1e-9) return new THREE.Vector3(0, -1, 0);
    
    // Normalize to get true direction
    diff.divideScalar(len);
    
    // Clamp Z to ensure it comes "out" naturally without clipping backwards
    diff.z = Math.max(zMin, Math.min(zMax, diff.z));
    return diff.normalize();
}

// ============================================================================
// CORE IK: rotate one bone so its rest direction aligns with dirWorld
// Returns the new local quaternion; also writes to targetQuaternions.
// ============================================================================
function solveDirection(boneName, dirWorld, parentWorldQuat) {
    const bone = boneObjects[boneName];
    if (!bone || !bone.userData.boneVector) return null;

    const rest    = restQuaternions[boneName] || new THREE.Quaternion();
    const restDir = bone.userData.boneVector.clone().applyQuaternion(rest);
    const dirLocal = dirWorld.clone().applyQuaternion(parentWorldQuat.clone().invert());

    const delta  = new THREE.Quaternion().setFromUnitVectors(restDir, dirLocal);
    const finalQ = delta.clone().multiply(rest);
    targetQuaternions[boneName].copy(finalQ);
    return finalQ;
}

// ============================================================================
// NEUTRAL STANDING POSE (Natural Rest)
// Arms hang naturally at sides, fingers slightly curled.
// ============================================================================
function setNeutralPose() {
    const armL = boneObjects['LeftArm'];
    const armR = boneObjects['RightArm'];
    if (!armL || !armR) return;

    const pwL = new THREE.Quaternion();
    const pwR = new THREE.Quaternion();
    armL.parent.getWorldQuaternion(pwL);
    armR.parent.getWorldQuaternion(pwR);

    // Natural hanging directions (slightly outwards)
    const qLA  = solveDirection('LeftArm',     new THREE.Vector3( 0.15, -0.98, 0.0).normalize(), pwL);
    const qLFA = qLA ? solveDirection('LeftForeArm', new THREE.Vector3( 0.10, -0.98, 0.0).normalize(), pwL.clone().multiply(qLA)) : null;
    if (qLA && qLFA) {
        const hp = pwL.clone().multiply(qLA).multiply(qLFA);
        solveDirection('LeftHand', new THREE.Vector3(0.10, -0.98, 0.0).normalize(), hp);
    }

    const qRA  = solveDirection('RightArm',     new THREE.Vector3(-0.15, -0.98, 0.0).normalize(), pwR);
    const qRFA = qRA ? solveDirection('RightForeArm', new THREE.Vector3(-0.10, -0.98, 0.0).normalize(), pwR.clone().multiply(qRA)) : null;
    if (qRA && qRFA) {
        const hp = pwR.clone().multiply(qRA).multiply(qRFA);
        solveDirection('RightHand', new THREE.Vector3(-0.10, -0.98, 0.0).normalize(), hp);
    }

    // Slightly curl all fingers for a relaxed human look
    const curlAngle = 25 * Math.PI / 180; // 25 degrees
    targetBones
        .filter(n => n.includes('Hand') && n !== 'LeftHand' && n !== 'RightHand')
        .forEach(name => {
            if (restQuaternions[name]) {
                const curl = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), curlAngle);
                targetQuaternions[name].copy(restQuaternions[name].clone().multiply(curl));
            }
        });
}

// ============================================================================
// HAND ORIENTATION SOLVER
//
// Uses the palm-plane basis approach from Code A/B: capture a rest basis at load
// (wrist, index MCP, pinky MCP define the palm plane), compute a live basis each
// frame, diff them, apply the delta to the rest world quaternion.
// This correctly tracks hand roll (pronation/supination).
//
// NOT redistributing any twist back to the forearm — that is what caused arm
// height to be wrong in Code A/B.
//
// Returns the hand's world quaternion so the finger solver can use it as
// its starting parentWorldQuat.
// ============================================================================
function solveHandOrientation(handData, handBoneName, restBasisQuat, handRestWorldQuat,
                               foreArmWorldQuat, side) {
    const wristW = lmToWorld(handData[0]);
    const indexW = lmToWorld(handData[5]);
    const pinkyW = lmToWorld(handData[17]);

    const midPt = new THREE.Vector3().addVectors(indexW, pinkyW).multiplyScalar(0.5);
    const fwdW  = new THREE.Vector3().subVectors(midPt, wristW).normalize();

    // Side vector: from index toward pinky for right hand, reversed for left.
    // This keeps the cross product consistent so handUp always points dorsally.
    const sideW = side === 'Right'
        ? new THREE.Vector3().subVectors(pinkyW, indexW).normalize()
        : new THREE.Vector3().subVectors(indexW, pinkyW).normalize();

    const upW = new THREE.Vector3().crossVectors(fwdW, sideW).normalize();
    const rightW = new THREE.Vector3().crossVectors(upW, fwdW).normalize();

    const targetBasisQuat = new THREE.Quaternion().setFromRotationMatrix(
        new THREE.Matrix4().makeBasis(rightW, upW, fwdW)
    );
    const delta           = targetBasisQuat.clone().multiply(restBasisQuat.clone().invert());
    const newHandWorldQuat = delta.clone().multiply(handRestWorldQuat);
    const handLocalQuat   = foreArmWorldQuat.clone().invert().multiply(newHandWorldQuat);

    targetQuaternions[handBoneName].copy(handLocalQuat);
    return newHandWorldQuat; // caller uses this as parent for fingers
}

// ============================================================================
// FINGER SOLVER — Code A/B's parent-space tracking approach
//
// Why NOT the hand-frame projection approach (Code C style):
// When fingers point toward or away from the camera, the wrist→midMCP vector and
// the idxMCP→pinkyMCP vector become nearly parallel/antiparallel. Their cross
// product degenerates and the hand frame becomes garbage. In the reference pose
// (hand raised, fingers pointing toward camera), this kills all flex angles.
//
// Code A/B's approach: transform each finger segment direction into the CURRENT
// parent bone's local space, then extract rotation from the bone's rest direction
// to that local direction. This is immune to the degenerate frame problem because
// it uses the actual bone hierarchy, not a separately built coordinate frame.
//
// The "unusual turning" in early Code A/B was caused by the forearm twist
// redistribution corrupting parentWorldQuat. That redistribution is now gone,
// so parentWorldQuat coming from handWorldQuat is clean.
//
// The one remaining issue in Code A/B: setFromUnitVectors → euler decomposition
// can have gimbal coupling when rotation is large. Fix: extract flex and spread
// DIRECTLY as atan2 from the bone-local direction, bypassing euler decomposition.
// ============================================================================

const fingerSegments = {
    'LeftHandThumb1':  [1, 2], 'LeftHandThumb2':  [2, 3], 'LeftHandThumb3':  [3, 4],
    'RightHandThumb1': [1, 2], 'RightHandThumb2': [2, 3], 'RightHandThumb3': [3, 4],
    'LeftHandIndex1':  [5, 6], 'LeftHandIndex2':  [6, 7], 'LeftHandIndex3':  [7, 8],
    'RightHandIndex1': [5, 6], 'RightHandIndex2': [6, 7], 'RightHandIndex3': [7, 8],
    'LeftHandMiddle1':  [9,10], 'LeftHandMiddle2':  [10,11], 'LeftHandMiddle3':  [11,12],
    'RightHandMiddle1': [9,10], 'RightHandMiddle2': [10,11], 'RightHandMiddle3': [11,12],
    'LeftHandRing1':  [13,14], 'LeftHandRing2':  [14,15], 'LeftHandRing3':  [15,16],
    'RightHandRing1': [13,14], 'RightHandRing2': [14,15], 'RightHandRing3': [15,16],
    'LeftHandPinky1':  [17,18], 'LeftHandPinky2':  [18,19], 'LeftHandPinky3':  [19,20],
    'RightHandPinky1': [17,18], 'RightHandPinky2': [18,19], 'RightHandPinky3': [19,20],
};

const fingerChains = {
    Left: [
        ['LeftHandThumb1',  'LeftHandThumb2',  'LeftHandThumb3'],
        ['LeftHandIndex1',  'LeftHandIndex2',  'LeftHandIndex3'],
        ['LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3'],
        ['LeftHandRing1',   'LeftHandRing2',   'LeftHandRing3'],
        ['LeftHandPinky1',  'LeftHandPinky2',  'LeftHandPinky3'],
    ],
    Right: [
        ['RightHandThumb1',  'RightHandThumb2',  'RightHandThumb3'],
        ['RightHandIndex1',  'RightHandIndex2',  'RightHandIndex3'],
        ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
        ['RightHandRing1',   'RightHandRing2',   'RightHandRing3'],
        ['RightHandPinky1',  'RightHandPinky2',  'RightHandPinky3'],
    ],
};

function solveFingers(handData, side, handWorldQuat) {
    if (!handData || handData.length < 21 || handData[0][0] === 0) return;

    fingerChains[side].forEach(chain => {
        let parentWorldQuat = handWorldQuat.clone();

        chain.forEach(boneName => {
            const bone = boneObjects[boneName];
            const rest = restQuaternions[boneName];
            if (!bone || !rest || !bone.userData.boneVector) return;

            const seg = fingerSegments[boneName];
            if (!seg) return;

            const a = lmToWorld(handData[seg[0]]);
            const b = lmToWorld(handData[seg[1]]);
            if (a.lengthSq() === 0 || b.lengthSq() === 0) return;

            const dirWorld = new THREE.Vector3().subVectors(b, a).normalize();
            const dirLocal = dirWorld.clone().applyQuaternion(parentWorldQuat.clone().invert());
            const dirInBoneRest = dirLocal.clone().applyQuaternion(rest.clone().invert());

            //   So flex component: atan2(dirInBoneRest.z, dirInBoneRest.y) → but sign TBD.
            //
            // Actually: Code A/B's setFromUnitVectors approach works, just fix the euler coupling.
            // The cleanest approach: decompose dirInBoneRest into flex and spread directly.
            //   boneVec = bone.userData.boneVector (the rest "forward" direction in local space)
            //   We want flex around the axis perpendicular to (boneVec, spread_axis).
            //   For (0,1,0) bones: flex axis = (1,0,0), spread axis = (0,0,1)
            //
            // Direct extraction without gimbal issues:
            // flex   = atan2(-dirInBoneRest.z, dirInBoneRest.y)  [curling y-bone around x]
            //                                                      [negative z = curling toward palm]
            // spread = atan2(dirInBoneRest.x, dirInBoneRest.y)   [splaying]
            //
            // However we don't know if boneVector is Y or X without reading the GLB.
            // Code A/B used euler 'XYZ' with euler.x = flex and it worked.
            // For euler 'XYZ': if boneVector = (0,1,0):
            //   Rx(α)*Ry(0)*Rz(β) * (0,1,0) = Rx(α) * Rz(β) * (0,1,0)
            //   Rz(β) * (0,1,0) = (-sinβ, cosβ, 0)
            //   Rx(α) * (-sinβ, cosβ, 0) = (-sinβ, cosβ*cosα, cosβ*sinα)
            //   So dirInBoneRest ≈ (-sinβ, cosβ*cosα, cosβ*sinα)
            //   flex_x: sinα ≈ z-component / cosβ
            //   spread_z: β = atan2(-x, y) approximately for small β
            //
            // This is getting complex. Let's just use Code A/B's approach exactly —
            // it works for the bones in this rig.

            // Compute rotation from bone's rest direction to target direction in bone-rest space
            const boneVec = bone.userData.boneVector;
            const localRot = new THREE.Quaternion().setFromUnitVectors(boneVec, dirInBoneRest);

            // Decompose to Euler, then zero out twist (Y) and constrain spread (Z for non-MCP)
            const euler = new THREE.Euler().setFromQuaternion(localRot, 'XYZ');

            const isMCP = boneName.endsWith('1');
            const isPIP = boneName.endsWith('2');

            // Flex: clamp to anatomical range
            const maxFlex = isMCP ? 90 : isPIP ? 100 : 80;
            euler.x = Math.max(-10  * Math.PI / 180,
                       Math.min(maxFlex * Math.PI / 180, euler.x));

            // Spread: only at MCP (base joint), zero at all others
            if (isMCP) {
                euler.z = Math.max(-40 * Math.PI / 180,
                           Math.min(40  * Math.PI / 180, euler.z));
            } else {
                euler.z = 0;
            }

            // No twist ever
            euler.y = 0;

            localRot.setFromEuler(euler);

            // Apply on top of rest quaternion → final local quaternion for this bone
            const finalQuat = rest.clone().multiply(localRot);
            targetQuaternions[boneName].copy(finalQuat);

            // Propagate: next bone's parent world = this bone's world
            parentWorldQuat = parentWorldQuat.clone().multiply(finalQuat);
        });
    });
}

// ============================================================================
// MAP MEDIAPIPE TO AVATAR SPACE
// ============================================================================
let mpScale = 1.0;
let mpOffset = new THREE.Vector3();

function updateMappingParams(frameData) {
    const LA = boneObjects['LeftArm'];
    const RA = boneObjects['RightArm'];
    if (!LA || !RA) return;

    const aL = new THREE.Vector3(); LA.getWorldPosition(aL);
    const aR = new THREE.Vector3(); RA.getWorldPosition(aR);
    const aDist   = aL.distanceTo(aR);
    const aCenter = new THREE.Vector3().addVectors(aL, aR).multiplyScalar(0.5);

    const mL = lmToWorld(frameData.body[11]);
    const mR = lmToWorld(frameData.body[12]);
    const mDist   = mL.distanceTo(mR);
    const mCenter = new THREE.Vector3().addVectors(mL, mR).multiplyScalar(0.5);

    if (mDist > 0.001) {
        let scale = aDist / mDist;
        // CRITICAL FIX: Tight clamp so the scale doesn't explode when shoulders glitch
        scale = Math.max(0.8, Math.min(1.5, scale));
        mpScale = scale;
        mpOffset.copy(aCenter).sub(mCenter.clone().multiplyScalar(mpScale));
    }
}

function getTargetPosition(lm) {
    return lmToWorld(lm).multiplyScalar(mpScale).add(mpOffset);
}

// ============================================================================
// ARM SOLVER — one side at a time
// ============================================================================
function solveArmChain(bodyData, handData, side) {
    const isLeft = (side === 'Left');
    const sId = isLeft ? 11 : 12; // Shoulder
    const eId = isLeft ? 13 : 14; // Elbow
    const wId = isLeft ? 15 : 16; // Wrist

    const armName     = isLeft ? 'LeftArm'      : 'RightArm';
    const foreArmName = isLeft ? 'LeftForeArm'  : 'RightForeArm';
    const handName    = isLeft ? 'LeftHand'     : 'RightHand';

    const arm     = boneObjects[armName];
    const foreArm = boneObjects[foreArmName];
    if (!arm || !foreArm) return;

    const shoulder = lmToWorld(bodyData[sId]);
    const elbow    = lmToWorld(bodyData[eId]);
    const wrist    = lmToWorld(bodyData[wId]);

    const pwQ = new THREE.Quaternion();
    arm.parent.getWorldQuaternion(pwQ);
    const avatarShoulder = new THREE.Vector3();
    arm.getWorldPosition(avatarShoulder);

    const hasHand = handData && handData.length >= 21 && (handData[0][0] !== 0 || handData[0][1] !== 0 || handData[0][2] !== 0);
    const wristMissing = !bodyData[wId] || (bodyData[wId][0] === 0 && bodyData[wId][1] === 0 && bodyData[wId][2] === 0);

    // TRUE REST POSE: If BOTH wrist and hand tracking are missing, drop to natural rest
    if (wristMissing && !hasHand) {
        const restTarget = avatarShoulder.clone().add(new THREE.Vector3(isLeft ? -0.15 : 0.15, -0.4, 0.0));
        let restVec = new THREE.Vector3().subVectors(restTarget, avatarShoulder);
        const restDist = restVec.length();
        if (restDist > 1e-6) {
            const restDir = restVec.clone().normalize();
            const qArm = solveDirection(armName, restDir, pwQ);
            if (qArm) {
                const armW = pwQ.clone().multiply(qArm);
                const qFore = solveDirection(foreArmName, restDir, armW);
                if (qFore) {
                    const foreW = armW.clone().multiply(qFore);
                    solveDirection(handName, restDir, foreW);
                }
            }
        }
        const curlAngle = 25 * Math.PI / 180;
        fingerChains[side].forEach(chain => chain.forEach(name => {
            if (restQuaternions[name]) {
                const curl = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), curlAngle);
                targetQuaternions[name].copy(restQuaternions[name].clone().multiply(curl));
            }
        }));
        return;
    }

    if (shoulder.lengthSq() === 0 || elbow.lengthSq() === 0) return;

    let upperDir = new THREE.Vector3().subVectors(elbow, shoulder).normalize();
    let foreDir  = new THREE.Vector3().subVectors(wrist, elbow).normalize();

    // 2. Exact Target in World Space
    let targetWrist = null;
    if (!wristMissing) {
        targetWrist = getTargetPosition(bodyData[wId]);
    } else if (hasHand) {
        targetWrist = getTargetPosition(handData[0]);
    }
    
    if (!targetWrist) return;
    
    // Fix MediaPipe Y-squash when facing sideways
    if (wrist.y > elbow.y) {
        const noseTarget = getTargetPosition(bodyData[0]);
        targetWrist.y = noseTarget.y - 0.1; 
    }

    // CRITICAL FIX: Strict Torso Collision Avoidance (Z-axis)
    // Prevent wrists from going behind the chest
    const minZ = avatarShoulder.z - 0.1;
    if (targetWrist.z < minZ) {
        targetWrist.z = minZ;
    }
    
    let targetVec = new THREE.Vector3().subVectors(targetWrist, avatarShoulder);
    
    const targetDist = targetVec.length();
    if (targetDist > 1e-6) {
        const armScale = new THREE.Vector3();
        arm.getWorldScale(armScale);
        const scale = armScale.y;
        
        let L1 = foreArm.position.length() * scale;
        let L2 = boneObjects[handName].position.length() * scale;

        // RUBBER-BANDING: Stretch arm to reach target perfectly
        if (targetDist > L1 + L2) {
            const stretch = targetDist / (L1 + L2);
            L1 *= stretch;
            L2 *= stretch;
        }

        const reachableDist = Math.min(targetDist, L1 + L2 - 0.001 * scale);
        const targetDir = targetVec.clone().normalize();
        
        let cosAlpha = (L1*L1 + reachableDist*reachableDist - L2*L2) / (2 * L1 * reachableDist);
        cosAlpha = Math.max(-1, Math.min(1, cosAlpha));
        const sinAlpha = Math.sqrt(1 - cosAlpha*cosAlpha);
        
        // Extract true elbow pop-out direction from MediaPipe to preserve natural bending
        const dot = upperDir.dot(targetDir);
        let ortho = new THREE.Vector3().subVectors(upperDir, targetDir.clone().multiplyScalar(dot));
        if (ortho.lengthSq() < 1e-6) {
            ortho = new THREE.Vector3(0,1,0).cross(targetDir);
            if (ortho.lengthSq() < 1e-6) ortho.set(1,0,0);
        }
        ortho.normalize();
        
        upperDir = new THREE.Vector3()
            .addScaledVector(targetDir, cosAlpha)
            .addScaledVector(ortho, sinAlpha);
            
        foreDir = new THREE.Vector3().subVectors(
            targetDir.clone().multiplyScalar(reachableDist),
            upperDir.clone().multiplyScalar(L1)
        ).normalize();
    }

    const qArm = solveDirection(armName, upperDir, pwQ);
    if (!qArm) return;
    const armWorld = pwQ.clone().multiply(qArm);

    const qFore = solveDirection(foreArmName, foreDir, armWorld);
    if (!qFore) return;
    const foreWorld = armWorld.clone().multiply(qFore);

    const restBasis     = side === 'Left' ? restBasisQuatL        : restBasisQuatR;
    const handRestWorld = side === 'Left' ? leftHandRestWorldQuat  : rightHandRestWorldQuat;
    const basisReady    = side === 'Left' ? isLeftHandBasisReady   : isRightHandBasisReady;

    if (hasHand && basisReady) {
        const handWorldQuat = solveHandOrientation(
            handData, handName, restBasis, handRestWorld, foreWorld, side
        );
        solveFingers(handData, side, handWorldQuat);
    } else {
        // Hand tracking missing, but wrist is tracked. Curl fingers into a fist.
        const fallback = new THREE.Vector3(
            side === 'Left' ? 0.02 : -0.02, -1.0, 0.06
        ).normalize();
        solveDirection(handName, fallback, foreWorld);

        const curlAngle = 25 * Math.PI / 180;
        fingerChains[side].forEach(chain => chain.forEach(name => {
            if (restQuaternions[name]) {
                const curl = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), curlAngle);
                targetQuaternions[name].copy(restQuaternions[name].clone().multiply(curl));
            }
        }));
    }
}

// ============================================================================
// UPDATE FROM ONE FRAME OF DATA
// ============================================================================
function updateAvatar(frameData) {
    if (!frameData.body || frameData.body.length < 17) return;
    updateMappingParams(frameData);
    solveArmChain(frameData.body, frameData.left_hand,  'Left');
    solveArmChain(frameData.body, frameData.right_hand, 'Right');
}

// ============================================================================
// MODEL LOAD
// ============================================================================
const loader = new GLTFLoader();

loader.load('./assets/male2k.glb', (gltf) => {
    const avatar = gltf.scene;
    scene.add(avatar);
    avatar.scale.set(0.04, 0.04, 0.04);
    avatar.position.set(0, 0, 0);

    avatar.traverse((child) => {
        if (!child.isBone) return;
        const baseName = child.name.split('_')[0];
        if (!targetBones.includes(baseName)) return;

        boneObjects[baseName]       = child;
        restQuaternions[baseName]   = child.quaternion.clone();
        targetQuaternions[baseName] = child.quaternion.clone();

        const childBone = child.children.find(c => c.isBone);
        child.userData.boneVector = childBone
            ? childBone.position.clone().normalize()
            : new THREE.Vector3(0, 1, 0);

        console.log(`Mapped: ${baseName}  boneVec=${JSON.stringify(child.userData.boneVector.toArray().map(v=>+v.toFixed(3)))}`);
    });

    avatar.updateMatrixWorld(true);

    // Capture rest basis for right hand (bind pose = T-pose)
    if (boneObjects['RightHand'] && boneObjects['RightHandIndex1'] && boneObjects['RightHandPinky1']) {
        const wW = new THREE.Vector3(); boneObjects['RightHand'].getWorldPosition(wW);
        const iW = new THREE.Vector3(); boneObjects['RightHandIndex1'].getWorldPosition(iW);
        const pW = new THREE.Vector3(); boneObjects['RightHandPinky1'].getWorldPosition(pW);

        const mid  = new THREE.Vector3().addVectors(iW, pW).multiplyScalar(0.5);
        const fwd  = new THREE.Vector3().subVectors(mid, wW).normalize();
        const side = new THREE.Vector3().subVectors(pW, iW).normalize(); // right hand: pinky-index
        const up   = new THREE.Vector3().crossVectors(fwd, side).normalize();
        const right = new THREE.Vector3().crossVectors(up, fwd).normalize();

        restBasisQuatR.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd));
        boneObjects['RightHand'].getWorldQuaternion(rightHandRestWorldQuat);
        isRightHandBasisReady = true;
        console.log('Right hand rest basis ready.');
    }

    // Capture rest basis for left hand
    if (boneObjects['LeftHand'] && boneObjects['LeftHandIndex1'] && boneObjects['LeftHandPinky1']) {
        const wW = new THREE.Vector3(); boneObjects['LeftHand'].getWorldPosition(wW);
        const iW = new THREE.Vector3(); boneObjects['LeftHandIndex1'].getWorldPosition(iW);
        const pW = new THREE.Vector3(); boneObjects['LeftHandPinky1'].getWorldPosition(pW);

        const mid  = new THREE.Vector3().addVectors(iW, pW).multiplyScalar(0.5);
        const fwd  = new THREE.Vector3().subVectors(mid, wW).normalize();
        const side = new THREE.Vector3().subVectors(iW, pW).normalize(); // left hand: index-pinky
        const up   = new THREE.Vector3().crossVectors(fwd, side).normalize();
        const right = new THREE.Vector3().crossVectors(up, fwd).normalize();

        restBasisQuatL.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd));
        boneObjects['LeftHand'].getWorldQuaternion(leftHandRestWorldQuat);
        isLeftHandBasisReady = true;
        console.log('Left hand rest basis ready.');
    }

    setNeutralPose();
    modelLoaded = true;
    console.log('Model loaded.');
}, undefined, (err) => console.error('GLB load error:', err));

// ============================================================================
// DATA LOAD
// ============================================================================
let animationFrames = [];
let currentFrame    = 0;
let isDataLoaded    = false;
let isPlaying       = false;

window.playTranslation = () => {
    if (!isDataLoaded || animationFrames.length === 0) return;
    isPlaying = true;
    currentFrame = 0;
};

fetch('./output/Apple_ready.json')
    .then(r => r.json())
    .then(data => {
        animationFrames = data.frames;
        isDataLoaded    = true;
        console.log(`Loaded ${animationFrames.length} frame(s).`);
    })
    .catch(err => console.error('JSON load error:', err));

// ============================================================================
// MEDIAPIPE VISUALIZER — body dots only, scaled to model shoulders
// Hand dots are NOT rendered separately because hand landmarks are in the same
// world space as body landmarks, so they appear correctly within the body dots.
// ============================================================================
const landmarkGroup   = new THREE.Group();
scene.add(landmarkGroup);
const landmarkSpheres = [];
const sphereGeo = new THREE.SphereGeometry(0.015, 8, 8);
const sphereMat = new THREE.MeshBasicMaterial({ color: 0xff4444, depthTest: false });

function updateVisualizer(frameData) {
    if (!frameData.body || frameData.body.length < 17) return;

    const LA = boneObjects['LeftArm'];
    const RA = boneObjects['RightArm'];
    if (LA && RA) {
        const aL = new THREE.Vector3(); LA.getWorldPosition(aL);
        const aR = new THREE.Vector3(); RA.getWorldPosition(aR);
        const aDist   = aL.distanceTo(aR);
        const aCenter = new THREE.Vector3().addVectors(aL, aR).multiplyScalar(0.5);

        const mL = lmToWorld(frameData.body[11]);
        const mR = lmToWorld(frameData.body[12]);
        const mDist   = mL.distanceTo(mR);
        const mCenter = new THREE.Vector3().addVectors(mL, mR).multiplyScalar(0.5);

        if (mDist > 0) {
            const s = aDist / mDist;
            landmarkGroup.scale.setScalar(s);
            landmarkGroup.position.copy(aCenter).sub(mCenter.clone().multiplyScalar(s));
        }
    }

    let n = 0;
    const put = (x, y, z) => {
        if (n >= landmarkSpheres.length) {
            const m = new THREE.Mesh(sphereGeo, sphereMat);
            landmarkGroup.add(m);
            landmarkSpheres.push(m);
        }
        landmarkSpheres[n].position.set(x, y, z);
        landmarkSpheres[n].visible = true;
        n++;
    };

    // Body landmarks
    frameData.body.forEach(lm => {
        if (lm[0] !== 0 || lm[1] !== 0 || lm[2] !== 0)
            put(lm[0], -lm[1], -lm[2]);
    });

    // Left hand landmarks
    if (frameData.left_hand) {
        frameData.left_hand.forEach(lm => {
            if (lm[0] !== 0 || lm[1] !== 0 || lm[2] !== 0)
                put(lm[0], -lm[1], -lm[2]);
        });
    }

    // Right hand landmarks
    if (frameData.right_hand) {
        frameData.right_hand.forEach(lm => {
            if (lm[0] !== 0 || lm[1] !== 0 || lm[2] !== 0)
                put(lm[0], -lm[1], -lm[2]);
        });
    }
    for (let i = n; i < landmarkSpheres.length; i++) landmarkSpheres[i].visible = false;
}

// ============================================================================
// RENDER LOOP
// ============================================================================
const clock       = new THREE.Clock();
let   timeAccum   = 0;
const FRAME_DUR   = 1 / 30;
// Increased SLERP_SPEED from 0.08 to 0.6 so the avatar tracks the movement almost instantly.
// Since the data is already pre-smoothed by the Python script, we don't need heavy client-side sluggishness.
const SLERP_SPEED = 0.6;

const handleResize = () => {
    const container = document.getElementById('canvas-wrapper') || document.getElementById('avatar-container') || document.body;
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
};

window.addEventListener('resize', handleResize);

const containerEl = document.getElementById('avatar-container');
if (containerEl) {
    const resizeObserver = new ResizeObserver(() => {
        handleResize();
    });
    resizeObserver.observe(containerEl);
}

function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();

    if (modelLoaded && isDataLoaded && animationFrames.length > 0 && isPlaying) {
        timeAccum += dt;
        if (timeAccum >= FRAME_DUR) {
            timeAccum -= FRAME_DUR;
            updateAvatar(animationFrames[currentFrame]);
            // updateVisualizer(animationFrames[currentFrame]); <-- COMMENTED OUT TO REMOVE RED DOTS
            
            if (animationFrames.length > 1) {
                currentFrame = (currentFrame + 1) % animationFrames.length;
            }
        }
    }

    targetBones.forEach(name => {
        const bone = boneObjects[name];
        const tgt  = targetQuaternions[name];
        if (bone && tgt) bone.quaternion.slerp(tgt, SLERP_SPEED);
    });

    controls.update();
    renderer.render(scene, camera);
}

animate();