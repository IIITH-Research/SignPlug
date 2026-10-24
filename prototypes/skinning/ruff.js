import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- 1. Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.5, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 4, 0);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(5, 5, 5);
scene.add(dirLight);
scene.add(new THREE.GridHelper(10, 10, 0x555555, 0x444444));

// --- 2. Load Avatar ---
const loader = new GLTFLoader();

// ALL bones present for both hands
const targetBones = [
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4',
    'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4',
    'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4',
    'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4',
    'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4',
    'RightArm', 'RightForeArm', 'RightHand',
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4',
    'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4',
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4',
    'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4',
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4'
];

const boneObjects = {};
const fingerRestQuaternions = {};
let modelLoaded = false;

let restBasisQuatR = new THREE.Quaternion();
let rightHandRestWorldQuat = new THREE.Quaternion();
let isRightHandBasisReady = false;

let restBasisQuatL = new THREE.Quaternion();
let leftHandRestWorldQuat = new THREE.Quaternion();
let isLeftHandBasisReady = false;

loader.load('./assets/male2k.glb', (gltf) => {
    const avatar = gltf.scene;
    scene.add(avatar);
    avatar.scale.set(0.04, 0.04, 0.04);
    avatar.position.set(0, 0, 0);

    avatar.traverse((child) => {
        if (child.isBone) {
            const baseName = child.name.split('_')[0];
            if (targetBones.includes(baseName)) {
                boneObjects[baseName] = child;
                fingerRestQuaternions[baseName] = child.quaternion.clone();
                const targetChildBone = child.children.find(c => c.isBone);
                child.userData.boneVector = targetChildBone
                    ? targetChildBone.position.clone().normalize()
                    : new THREE.Vector3(0, 1, 0);
                console.log(`Mapped: ${baseName}`);
            }
        }
    });

    avatar.updateMatrixWorld(true);

    // --- RIGHT HAND REST BASIS ---
    if (boneObjects['RightHand'] && boneObjects['RightHandIndex1'] && boneObjects['RightHandPinky1']) {
        const wristW = new THREE.Vector3(); boneObjects['RightHand'].getWorldPosition(wristW);
        const indexW = new THREE.Vector3(); boneObjects['RightHandIndex1'].getWorldPosition(indexW);
        const pinkyW = new THREE.Vector3(); boneObjects['RightHandPinky1'].getWorldPosition(pinkyW);

        const midW    = new THREE.Vector3().addVectors(indexW, pinkyW).multiplyScalar(0.5);
        const fwdW    = new THREE.Vector3().subVectors(midW, wristW).normalize();
        const rightW  = new THREE.Vector3().subVectors(pinkyW, indexW).normalize(); // pinky - index for RIGHT hand
        const upW     = new THREE.Vector3().crossVectors(fwdW, rightW).normalize();
        rightW.crossVectors(upW, fwdW).normalize();

        restBasisQuatR.setFromRotationMatrix(new THREE.Matrix4().makeBasis(rightW, upW, fwdW));
        boneObjects['RightHand'].getWorldQuaternion(rightHandRestWorldQuat);
        isRightHandBasisReady = true;
        console.log("Right hand rest basis ready.");
    }

    // --- LEFT HAND REST BASIS ---
    if (boneObjects['LeftHand'] && boneObjects['LeftHandIndex1'] && boneObjects['LeftHandPinky1']) {
        const wristW = new THREE.Vector3(); boneObjects['LeftHand'].getWorldPosition(wristW);
        const indexW = new THREE.Vector3(); boneObjects['LeftHandIndex1'].getWorldPosition(indexW);
        const pinkyW = new THREE.Vector3(); boneObjects['LeftHandPinky1'].getWorldPosition(pinkyW);

        const midW    = new THREE.Vector3().addVectors(indexW, pinkyW).multiplyScalar(0.5);
        const fwdW    = new THREE.Vector3().subVectors(midW, wristW).normalize();
        const rightW  = new THREE.Vector3().subVectors(pinkyW, indexW).normalize(); // SAME as right hand
        const upW     = new THREE.Vector3().crossVectors(fwdW, rightW).normalize();
        rightW.crossVectors(upW, fwdW).normalize();

        restBasisQuatL.setFromRotationMatrix(new THREE.Matrix4().makeBasis(rightW, upW, fwdW));
        boneObjects['LeftHand'].getWorldQuaternion(leftHandRestWorldQuat);
        isLeftHandBasisReady = true;
        console.log("Left hand rest basis ready.");
    }

    modelLoaded = true;
    console.log("Model loaded successfully!");
}, undefined, (error) => console.error("Error loading GLB:", error));

const targetQuaternions = {};
targetBones.forEach(bone => { targetQuaternions[bone] = new THREE.Quaternion(); });

// --- Swing-Twist Decomposition ---
function decomposeSwingTwist(q, twistAxis) {
    const p = new THREE.Vector3(q.x, q.y, q.z);
    const proj = new THREE.Vector3().copy(twistAxis).multiplyScalar(p.dot(twistAxis));
    let twist = new THREE.Quaternion(proj.x, proj.y, proj.z, q.w);
    twist = (twist.lengthSq() === 0) ? new THREE.Quaternion() : twist.normalize();
    const swing = twist.clone().invert().multiply(q);
    return { swing, twist };
}

// --- Generic single-hand solver ---
function solveHandIK(
    handData,
    handBone,
    restBasisQuat,
    handRestWorldQuat,
    futureForeArmWorldQuat,
    futureArmWorldQuat,
    foreArmLocalQuat,
    foreArmBoneName,
    fingerChains,
    fingerIndices,
    sign
) {
    const v = (lm) => new THREE.Vector3(sign * lm[0], -lm[1], -lm[2]);

    const wristWorld = v(handData[0]);
    const indexWorld = v(handData[5]);
    const pinkyWorld = v(handData[17]);

    const midPt  = new THREE.Vector3().addVectors(indexWorld, pinkyWorld).multiplyScalar(0.5);
    const fwdW   = new THREE.Vector3().subVectors(midPt, wristWorld).normalize();
    const rightW = new THREE.Vector3().subVectors(pinkyWorld, indexWorld).normalize();
    const upW    = new THREE.Vector3().crossVectors(fwdW, rightW).normalize();
    rightW.crossVectors(upW, fwdW).normalize();

    const targetBasisQuat  = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(rightW, upW, fwdW));
    const deltaQuat        = targetBasisQuat.multiply(restBasisQuat.clone().invert());
    const newHandWorldQuat = deltaQuat.multiply(handRestWorldQuat);
    const rawHandLocal     = futureForeArmWorldQuat.clone().invert().multiply(newHandWorldQuat);

    const handRestLocal = fingerRestQuaternions[handBone.name.split('_')[0]] || new THREE.Quaternion();
    const relRot        = handRestLocal.clone().invert().multiply(rawHandLocal);
    const twistAxis     = handBone.userData.boneVector.clone().normalize();

    const { swing, twist } = decomposeSwingTwist(relRot, twistAxis);

    const twistInForeArmSpace = handRestLocal.clone().multiply(twist).multiply(handRestLocal.clone().invert());
    
    // Project the twist onto the ForeArm's actual bone axis to prevent it from swinging the elbow!
    const foreArmBone = boneObjects[foreArmBoneName];
    const foreArmAxis = foreArmBone.userData.boneVector.clone().normalize();
    const { twist: pureForeArmTwist } = decomposeSwingTwist(twistInForeArmSpace, foreArmAxis);

    foreArmLocalQuat.multiply(pureForeArmTwist);
    targetQuaternions[foreArmBoneName].copy(foreArmLocalQuat);
    const updatedForeArmWorldQuat = futureArmWorldQuat.clone().multiply(foreArmLocalQuat);

    const finalHandLocal = handRestLocal.clone().multiply(swing);
    const handBoneName   = handBone.name.split('_')[0];
    targetQuaternions[handBoneName].copy(finalHandLocal);

    // --- Fingers ---
    fingerChains.forEach((chain) => {
        // FIX: Compute future world rotation instead of reading stale scene state!
        let parentWorldQuat = updatedForeArmWorldQuat.clone().multiply(finalHandLocal);

        chain.forEach(boneName => {
            const bone = boneObjects[boneName];
            if (!bone || !bone.userData.boneVector) return;

            const idxs     = fingerIndices[boneName];
            const startVec = v(handData[idxs[0]]);
            const endVec   = v(handData[idxs[1]]);
            if (startVec.length() === 0 || endVec.length() === 0) return;

            const dirWorld = new THREE.Vector3().subVectors(endVec, startVec).normalize();
            
            const restQuat = fingerRestQuaternions[boneName] || new THREE.Quaternion();
            const dirLocal = dirWorld.clone().applyQuaternion(parentWorldQuat.clone().invert());
            
            // Target direction mapped into the bone's REST space
            const targetDirBone = dirLocal.clone().applyQuaternion(restQuat.clone().invert());
            
            // Shortest rotation from the bone's local axis (+Y) to the target direction
            const localRot = new THREE.Quaternion().setFromUnitVectors(bone.userData.boneVector, targetDirBone);
            
            const euler = new THREE.Euler().setFromQuaternion(localRot, 'XYZ');
            
            const isMCP = boneName.endsWith('1');
            const isPIP = boneName.endsWith('2');
            const isDIP = boneName.endsWith('3');
            
            // Flexion/Extension Limits (Pitch)
            const pitchLimit = isMCP ? 100 * Math.PI / 180
                             : isPIP ? 95  * Math.PI / 180
                             : isDIP ? 90  * Math.PI / 180 : 100 * Math.PI / 180;
                             
            euler.x = Math.max(-pitchLimit, Math.min(pitchLimit, euler.x));
            
            // Spread Limits (Yaw)
            if (isMCP) {
                // Base joints (including thumb) can spread widely
                const yawLimit = 45 * Math.PI / 180;
                euler.z = Math.max(-yawLimit, Math.min(yawLimit, euler.z));
            } else {
                // Middle and tip joints are pure hinges (no side-to-side bending)
                euler.z = 0;
            }
            
            // Twist Limits (Roll) - Fingers do not twist
            euler.y = 0;
            
            localRot.setFromEuler(euler);
            
            // Apply computed local rotation on top of the original rest pose
            const finalQuat = restQuat.clone().multiply(localRot);
            
            targetQuaternions[boneName].copy(finalQuat);
            parentWorldQuat.multiply(finalQuat);
        });
    });

    return updatedForeArmWorldQuat;
}

// --- 3. Main IK Solver ---
function updateArmsIK(frameData) {
    const body = frameData.body;
    if (!body || body.length < 17) return;

    // LEFT ARM
    const leftArm     = boneObjects['LeftArm'];
    const leftForeArm = boneObjects['LeftForeArm'];

    if (leftArm && leftForeArm && leftArm.parent && leftArm.userData.boneVector) {
        const startArmL = new THREE.Vector3(body[11][0], -body[11][1], -body[11][2]);
        const endArmL   = new THREE.Vector3(body[13][0], -body[13][1], -body[13][2]);

        if (startArmL.length() > 0 && endArmL.length() > 0) {
            const parentWorldQuatL = new THREE.Quaternion();
            leftArm.parent.getWorldQuaternion(parentWorldQuatL);

            const dirArmL      = new THREE.Vector3().subVectors(endArmL, startArmL).normalize();
            const dirArmLocalL = dirArmL.clone().applyQuaternion(parentWorldQuatL.clone().invert());
            
            const restArmL     = fingerRestQuaternions['LeftArm'] || new THREE.Quaternion();
            const restDirArmL  = leftArm.userData.boneVector.clone().applyQuaternion(restArmL);
            const deltaArmL    = new THREE.Quaternion().setFromUnitVectors(restDirArmL, dirArmLocalL);
            const quatArmL     = deltaArmL.clone().multiply(restArmL);
            
            targetQuaternions['LeftArm'].copy(quatArmL);

            const endForeArmL       = new THREE.Vector3(body[15][0], -body[15][1], -body[15][2]);
            const futureArmWorldL   = parentWorldQuatL.clone().multiply(quatArmL);

            const dirForeL      = new THREE.Vector3().subVectors(endForeArmL, endArmL).normalize();
            const dirForeLocalL = dirForeL.clone().applyQuaternion(futureArmWorldL.clone().invert());
            
            const restForeL     = fingerRestQuaternions['LeftForeArm'] || new THREE.Quaternion();
            const restDirForeL  = leftForeArm.userData.boneVector.clone().applyQuaternion(restForeL);
            const deltaForeL    = new THREE.Quaternion().setFromUnitVectors(restDirForeL, dirForeLocalL);
            let quatForeArmL    = deltaForeL.clone().multiply(restForeL);
            
            targetQuaternions['LeftForeArm'].copy(quatForeArmL);

            let futureForeArmWorldL = futureArmWorldL.clone().multiply(quatForeArmL);

            const leftHandData = frameData.left_hand;
            const leftHandBone = boneObjects['LeftHand'];

            if (leftHandData && leftHandData[0][0] !== 0 && leftHandBone && isLeftHandBasisReady) {
                const fingerChainsL = [
                    ['LeftHandThumb1',  'LeftHandThumb2',  'LeftHandThumb3'],
                    ['LeftHandIndex1',  'LeftHandIndex2',  'LeftHandIndex3'],
                    ['LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3'],
                    ['LeftHandRing1',   'LeftHandRing2',   'LeftHandRing3'],
                    ['LeftHandPinky1',  'LeftHandPinky2',  'LeftHandPinky3']
                ];
                const fingerIndicesL = {
                    'LeftHandThumb1':  [1,2],  'LeftHandThumb2':  [2,3],  'LeftHandThumb3':  [3,4],
                    'LeftHandIndex1':  [5,6],  'LeftHandIndex2':  [6,7],  'LeftHandIndex3':  [7,8],
                    'LeftHandMiddle1': [9,10], 'LeftHandMiddle2': [10,11],'LeftHandMiddle3': [11,12],
                    'LeftHandRing1':   [13,14],'LeftHandRing2':   [14,15],'LeftHandRing3':   [15,16],
                    'LeftHandPinky1':  [17,18],'LeftHandPinky2':  [18,19],'LeftHandPinky3':  [19,20]
                };

                solveHandIK(
                    leftHandData, leftHandBone,
                    restBasisQuatL, leftHandRestWorldQuat,
                    futureForeArmWorldL, futureArmWorldL,
                    quatForeArmL, 'LeftForeArm',
                    fingerChainsL, fingerIndicesL,
                    +1
                );
            }
        }
    }

    // RIGHT ARM
    const rightArm     = boneObjects['RightArm'];
    const rightForeArm = boneObjects['RightForeArm'];

    if (rightArm && rightForeArm && rightArm.parent && rightArm.userData.boneVector) {
        const startArmR = new THREE.Vector3(body[12][0], -body[12][1], -body[12][2]);
        const endArmR   = new THREE.Vector3(body[14][0], -body[14][1], -body[14][2]);

        if (startArmR.length() > 0 && endArmR.length() > 0) {
            const parentWorldQuatR = new THREE.Quaternion();
            rightArm.parent.getWorldQuaternion(parentWorldQuatR);

            const dirArmR      = new THREE.Vector3().subVectors(endArmR, startArmR).normalize();
            const dirArmLocalR = dirArmR.clone().applyQuaternion(parentWorldQuatR.clone().invert());
            
            const restArmR     = fingerRestQuaternions['RightArm'] || new THREE.Quaternion();
            const restDirArmR  = rightArm.userData.boneVector.clone().applyQuaternion(restArmR);
            const deltaArmR    = new THREE.Quaternion().setFromUnitVectors(restDirArmR, dirArmLocalR);
            const quatArmR     = deltaArmR.clone().multiply(restArmR);
            
            targetQuaternions['RightArm'].copy(quatArmR);

            const endForeArmR       = new THREE.Vector3(body[16][0], -body[16][1], -body[16][2]);
            const futureArmWorldR   = parentWorldQuatR.clone().multiply(quatArmR);

            const dirForeR      = new THREE.Vector3().subVectors(endForeArmR, endArmR).normalize();
            const dirForeLocalR = dirForeR.clone().applyQuaternion(futureArmWorldR.clone().invert());
            
            const restForeR     = fingerRestQuaternions['RightForeArm'] || new THREE.Quaternion();
            const restDirForeR  = rightForeArm.userData.boneVector.clone().applyQuaternion(restForeR);
            const deltaForeR    = new THREE.Quaternion().setFromUnitVectors(restDirForeR, dirForeLocalR);
            let quatForeArmR    = deltaForeR.clone().multiply(restForeR);
            
            targetQuaternions['RightForeArm'].copy(quatForeArmR);

            let futureForeArmWorldR = futureArmWorldR.clone().multiply(quatForeArmR);

            const rightHandData = frameData.right_hand;
            const rightHandBone = boneObjects['RightHand'];

            if (rightHandData && rightHandData[0][0] !== 0 && rightHandBone && isRightHandBasisReady) {
                const fingerChainsR = [
                    ['RightHandThumb1',  'RightHandThumb2',  'RightHandThumb3'],
                    ['RightHandIndex1',  'RightHandIndex2',  'RightHandIndex3'],
                    ['RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3'],
                    ['RightHandRing1',   'RightHandRing2',   'RightHandRing3'],
                    ['RightHandPinky1',  'RightHandPinky2',  'RightHandPinky3']
                ];
                const fingerIndicesR = {
                    'RightHandThumb1':  [1,2],  'RightHandThumb2':  [2,3],  'RightHandThumb3':  [3,4],
                    'RightHandIndex1':  [5,6],  'RightHandIndex2':  [6,7],  'RightHandIndex3':  [7,8],
                    'RightHandMiddle1': [9,10], 'RightHandMiddle2': [10,11],'RightHandMiddle3': [11,12],
                    'RightHandRing1':   [13,14],'RightHandRing2':   [14,15],'RightHandRing3':   [15,16],
                    'RightHandPinky1':  [17,18],'RightHandPinky2':  [18,19],'RightHandPinky3':  [19,20]
                };

                solveHandIK(
                    rightHandData, rightHandBone,
                    restBasisQuatR, rightHandRestWorldQuat,
                    futureForeArmWorldR, futureArmWorldR,
                    quatForeArmR, 'RightForeArm',
                    fingerChainsR, fingerIndicesR,
                    +1
                );
            }
        }
    }
}

// --- 4. Load Animation Data ---
let animationFrames = [];
let currentFrame    = 0;
let isDataLoaded    = false;

// MediaPipe Visualizer
const landmarkGroup = new THREE.Group();
scene.add(landmarkGroup);
const landmarkSpheres = [];
const sphereGeo = new THREE.SphereGeometry(0.015, 8, 8);
const sphereMat = new THREE.MeshBasicMaterial({ color: 0xff0000, depthTest: false }); // Red, always visible

function updateMediaPipeVisualizer(frameData) {
    if (!frameData.body || frameData.body.length < 17) return;

    // Dynamically align the Visualizer to the Avatar's shoulders
    const leftArm = boneObjects['LeftArm'];
    const rightArm = boneObjects['RightArm'];
    if (leftArm && rightArm) {
        const avatarL = new THREE.Vector3(); leftArm.getWorldPosition(avatarL);
        const avatarR = new THREE.Vector3(); rightArm.getWorldPosition(avatarR);
        const avatarDist = avatarL.distanceTo(avatarR);
        const avatarCenter = new THREE.Vector3().addVectors(avatarL, avatarR).multiplyScalar(0.5);

        const mpL = new THREE.Vector3(frameData.body[11][0], -frameData.body[11][1], -frameData.body[11][2]);
        const mpR = new THREE.Vector3(frameData.body[12][0], -frameData.body[12][1], -frameData.body[12][2]);
        const mpDist = mpL.distanceTo(mpR);
        const mpCenter = new THREE.Vector3().addVectors(mpL, mpR).multiplyScalar(0.5);

        if (mpDist > 0) {
            const scale = avatarDist / mpDist;
            landmarkGroup.scale.set(scale, scale, scale);
            const scaledMpCenter = mpCenter.clone().multiplyScalar(scale);
            landmarkGroup.position.copy(avatarCenter).sub(scaledMpCenter);
        }
    }

    let pointCount = 0;
    const setSphere = (x, y, z) => {
        if (pointCount >= landmarkSpheres.length) {
            const mesh = new THREE.Mesh(sphereGeo, sphereMat);
            landmarkGroup.add(mesh);
            landmarkSpheres.push(mesh);
        }
        landmarkSpheres[pointCount].position.set(x, y, z);
        landmarkSpheres[pointCount].visible = true;
        pointCount++;
    };
    
    if (frameData.body) frameData.body.forEach(lm => { if (lm[0]!==0||lm[1]!==0||lm[2]!==0) setSphere(lm[0], -lm[1], -lm[2]); });
    if (frameData.left_hand) frameData.left_hand.forEach(lm => { if (lm[0]!==0||lm[1]!==0||lm[2]!==0) setSphere(lm[0], -lm[1], -lm[2]); });
    if (frameData.right_hand) frameData.right_hand.forEach(lm => { if (lm[0]!==0||lm[1]!==0||lm[2]!==0) setSphere(lm[0], -lm[1], -lm[2]); });
    
    for (let i = pointCount; i < landmarkSpheres.length; i++) landmarkSpheres[i].visible = false;
}

const JSON_FILE_PATH = './output/image2_ready.json';
fetch(JSON_FILE_PATH)
    .then(r => r.json())
    .then(data => {
        animationFrames = data.frames;
        console.log(`Loaded ${animationFrames.length} frames.`);
        isDataLoaded = true;
    })
    .catch(err => console.error("Error loading JSON:", err));

// --- 5. Render Loop ---
const clock        = new THREE.Clock();
let timeAccumulator = 0;
const FPS           = 30;
const frameDuration = 1.0 / FPS;
const SLERP_SPEED   = 0.05;

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    const deltaTime = clock.getDelta();

    if (modelLoaded && isDataLoaded && animationFrames.length > 0) {
        timeAccumulator += deltaTime;
        if (timeAccumulator >= frameDuration) {
            timeAccumulator -= frameDuration;
            updateArmsIK(animationFrames[currentFrame]);
            updateMediaPipeVisualizer(animationFrames[currentFrame]);
            if (animationFrames.length > 1) {
                currentFrame = (currentFrame + 1) % animationFrames.length;
            }
        }
        targetBones.forEach(boneName => {
            if (boneObjects[boneName]) {
                boneObjects[boneName].quaternion.slerp(targetQuaternions[boneName], SLERP_SPEED);
            }
        });
    }

    renderer.render(scene, camera);
}

animate();