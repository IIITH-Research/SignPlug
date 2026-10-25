import * as THREE from 'three';

export function solve2BoneIK(shoulderW, targetW, poleW, L1, L2) {
    const dirTarget = new THREE.Vector3().subVectors(targetW, shoulderW);
    const targetDist = dirTarget.length();
    
    if (targetDist === 0) return { dirArm: new THREE.Vector3(0,-1,0), dirFore: new THREE.Vector3(0,-1,0) };
    
    dirTarget.normalize();
    
    // Law of cosines to find the angle at the shoulder
    let cosAngle = (L1*L1 + targetDist*targetDist - L2*L2) / (2 * L1 * targetDist);
    cosAngle = Math.max(-1, Math.min(1, cosAngle));
    const shoulderAngle = Math.acos(cosAngle);
    
    // Find the bend plane normal (shoulder, target, pole)
    let dirPole = new THREE.Vector3().subVectors(poleW, shoulderW);
    let bendAxis = new THREE.Vector3().crossVectors(dirTarget, dirPole);
    
    if (bendAxis.lengthSq() < 1e-5) {
        // Fallback if pole is collinear
        bendAxis.set(0, 0, 1);
    } else {
        bendAxis.normalize();
    }
    
    // Rotate dirTarget around bendAxis by shoulderAngle to get the upper arm direction!
    const dirArm = dirTarget.clone().applyAxisAngle(bendAxis, shoulderAngle);
    
    // Virtual elbow position
    const elbowW = shoulderW.clone().add(dirArm.clone().multiplyScalar(L1));
    
    // Forearm direction
    const dirFore = new THREE.Vector3().subVectors(targetW, elbowW).normalize();
    
    return { dirArm, dirFore };
}
