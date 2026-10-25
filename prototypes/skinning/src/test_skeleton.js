import fs from 'fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// We need a DOM-less environment for GLTFLoader if we run it in node, 
// or I can just use a simple HTML file to log it to the console.
