import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ============================================================================
// SCENE SETUP
// ============================================================================
const scene = new THREE.Scene();
const container = document.getElementById('dataset-canvas-wrapper') || document.getElementById('canvas-wrapper') || document.body;
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

const boneObjects = {};
const restQuaternions = {};
const targetQuaternions = {};

let modelLoaded = false;
let isDataLoaded = false;
let animationFrames = [];
let currentFrame = 0;
let isPlaying = false;
let timeAccum = 0;
const FRAME_DUR = 1 / 30;

let currentPlayingName = null;

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

        boneObjects[baseName] = child;
        restQuaternions[baseName] = child.quaternion.clone();
        targetQuaternions[baseName] = child.quaternion.clone();
    });

    modelLoaded = true;
    console.log("Avatar loaded successfully!");
});

function applyFrame(frameIndex) {
    const frameData = animationFrames[frameIndex];
    if (!frameData) return;
    targetBones.forEach(name => {
        if (frameData[name]) {
            const bone = boneObjects[name];
            if (bone) {
                // Apply the skeletal position and rotation from the animation frame
                if (frameData[name].position) bone.position.fromArray(frameData[name].position);
                if (frameData[name].rotation) bone.quaternion.fromArray(frameData[name].rotation);
            }
        }
    });
}

// ============================================================================
// ANIMATION LOADER
// ============================================================================
async function loadSentenceAnimation(fileName, autoPlay = false) {
    try {
        const response = await fetch(`./intepolation_code/keyframes_interp/${fileName}.json`);
        if (response.ok) {
            const data = await response.json();
            animationFrames = data.frames || [];
            isDataLoaded = animationFrames.length > 0;
            currentFrame = 0;
            timeAccum = 0;
            currentPlayingName = fileName;
            
            console.log(`Loaded ${animationFrames.length} frames for ${fileName}`);
            
            if (isDataLoaded) {
                applyFrame(0);
                const slider = document.getElementById('timeline-slider');
                if (slider) {
                    slider.max = animationFrames.length - 1;
                    slider.value = 0;
                }
                if (autoPlay) {
                    isPlaying = true;
                    window.dispatchEvent(new CustomEvent('animationStarted', { detail: { name: currentPlayingName } }));
                }
            }
        } else {
            console.error(`Failed to load animation file: ./intepolation_code/keyframes_interp/${fileName}.json`);
        }
    } catch(e) {
        console.error("Error loading animation:", e);
    }
}

// Expose functions globally for the HTML UI
window.loadDatasetAnimation = loadSentenceAnimation;
window.getCurrentDatasetPlaybackState = () => ({
    isPlaying,
    currentPlayingName,
    currentFrame,
    totalFrames: animationFrames.length
});

window.toggleDatasetPlayPause = () => {
    if (animationFrames.length > 0) {
        isPlaying = !isPlaying;
        if (isPlaying) {
            window.dispatchEvent(new CustomEvent('animationStarted', { detail: { name: currentPlayingName } }));
        } else {
            window.dispatchEvent(new CustomEvent('animationPaused', { detail: { name: currentPlayingName } }));
        }
    }
    return isPlaying;
};

window.stopDatasetPlayback = () => {
    isPlaying = false;
    currentFrame = 0;
    applyFrame(0);
};

// ============================================================================
// RENDER & PLAYBACK LOOP
// ============================================================================
const clock = new THREE.Clock();

function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();

    if (modelLoaded && isDataLoaded && animationFrames.length > 0 && isPlaying) {
        timeAccum += dt;
        if (timeAccum >= FRAME_DUR) {
            timeAccum -= FRAME_DUR;
            
            if (currentFrame < animationFrames.length - 1) {
                currentFrame++;
                applyFrame(currentFrame);
                const slider = document.getElementById('timeline-slider');
                if (slider) {
                    slider.value = currentFrame;
                }
            } else {
                // Animation reached the end!
                isPlaying = false;
                // Dispatch event to update play buttons in UI
                window.dispatchEvent(new CustomEvent('animationFinished', { detail: { name: currentPlayingName } }));
            }
        }
    }

    controls.update();
    renderer.render(scene, camera);
}

animate();

const handleResize = () => {
    const container = document.getElementById('dataset-canvas-wrapper') || document.getElementById('canvas-wrapper') || document.body;
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
};

window.addEventListener('resize', handleResize);

// Setup slider input listener
document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('timeline-slider');
    if (slider) {
        slider.addEventListener('input', (e) => {
            isPlaying = false;
            currentFrame = parseInt(e.target.value);
            applyFrame(currentFrame);
            // Dispatch event to update play button icon
            window.dispatchEvent(new CustomEvent('animationPaused', { detail: { name: currentPlayingName } }));
        });
    }
});
