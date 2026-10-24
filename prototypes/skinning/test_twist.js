const THREE = require('three');

function decomposeSwingTwist(q, twistAxis) {
    const p = new THREE.Vector3(q.x, q.y, q.z);
    const proj = new THREE.Vector3().copy(twistAxis).multiplyScalar(p.dot(twistAxis));
    let twist = new THREE.Quaternion(proj.x, proj.y, proj.z, q.w);
    twist = (twist.lengthSq() === 0) ? new THREE.Quaternion() : twist.normalize();
    const swing = twist.clone().invert().multiply(q);
    return { swing, twist };
}

// Test case
const twistAxis = new THREE.Vector3(0, 1, 0);
const rot = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.5, 1.0, 0, 'YXZ'));
const { swing, twist } = decomposeSwingTwist(rot, twistAxis);
console.log("Swing:", new THREE.Euler().setFromQuaternion(swing).toArray().map(x=>x.toFixed(3)));
console.log("Twist:", new THREE.Euler().setFromQuaternion(twist).toArray().map(x=>x.toFixed(3)));
