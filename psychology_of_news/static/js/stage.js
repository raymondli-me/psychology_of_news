
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class StageController {
    constructor() {
        this.container = document.getElementById('container');
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });

        this.pointsData = [];
        this.pointMeshes = [];
        this.currentModel = 'gpt';
        this.labelSprites = [];
        this.clusterStats = {};
        this.topicNames = {};

        this.init();
    }

    init() {
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));

        // Resize handler
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });

        this.animate();
        this.fetchData();
        this.setupInteractions();
    }

    async fetchData() {
        try {
            const res = await fetch('/api/data');
            const data = await res.json();

            this.pointsData = data.points;
            this.clusterStats = data.cluster_stats;
            this.topicNames = data.topic_names;

            this.buildScene();
        } catch (e) {
            console.error("Failed to load data:", e);
            document.querySelector('.message.ai').textContent = "Error: Could not load visualization data. Is the server running?";
        }
    }

    buildScene() {
        // Find centroid
        let cx = 0, cy = 0, cz = 0;
        this.pointsData.forEach(p => {
            cx += p.x || 0;
            cy += p.y || 0;
            cz += p.z || 0;
        });
        cx /= this.pointsData.length;
        cy /= this.pointsData.length;
        cz /= this.pointsData.length;
        this.centroid = { x: cx, y: cy, z: cz };

        this.camera.position.set(cx, cy, cz + 12);
        this.controls.target.set(cx, cy, cz);

        // Create points
        const geometry = new THREE.SphereGeometry(0.1, 16, 16);

        this.pointsData.forEach((p, idx) => {
            // Reconstruct coordinates if they were flattened or named differently
            // Assuming visualizer saved them as x,y,z in the dict
            const x = p.x || 0;
            const y = p.y || 0;
            const z = p.z || 0;

            const color = this.getColorForPoint(p, this.currentModel);
            const material = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.4 });
            const mesh = new THREE.Mesh(geometry, material);

            mesh.position.set(x, y, z);
            mesh.userData = { ...p, index: idx, defaultColor: color };

            this.scene.add(mesh);
            this.pointMeshes.push(mesh);
        });

        this.createLabels();
    }

    getColorForPoint(point, model) {
        let score = 5;
        if (point[model + '_score'] !== undefined) score = point[model + '_score'];
        else if (point.mean_score !== undefined) score = point.mean_score;

        const t = (score - 1) / 9;
        let r, g, b;
        // Blue-Yellow-Red gradient
        if (t < 0.5) {
            r = Math.floor(59 + t * 2 * 175);
            g = Math.floor(130 + t * 2 * 67);
            b = Math.floor(246 - t * 2 * 152);
        } else {
            r = Math.floor(234 + (t - 0.5) * 2 * 5);
            g = Math.floor(179 - (t - 0.5) * 2 * 111);
            b = Math.floor(8 + (t - 0.5) * 2 * 60);
        }
        return new THREE.Color(`rgb(${r}, ${g}, ${b})`);
    }

    createLabels() {
        // Clear old
        this.labelSprites.forEach(s => this.scene.remove(s));
        this.labelSprites = [];

        // ... Logic to create labels similar to visualizer.py ...
        // Simplified for now: just place labels at centroids
        if (!this.clusterStats) return;

        for (const [clusterId, stats] of Object.entries(this.clusterStats)) {
            if (parseInt(clusterId) === -1 || stats.count < 3) continue;

            // Get name
            const modelKey = this.currentModel; // 'gpt', 'claude'
            const name = this.topicNames?.[modelKey]?.[clusterId] || `Cluster ${clusterId}`;

            this.createLabelSprite(name, stats.centroid);
        }
    }

    createLabelSprite(text, position) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = 'bold 32px Arial';
        const textWidth = ctx.measureText(text).width;
        canvas.width = textWidth + 20;
        canvas.height = 50;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.roundRect(0, 0, canvas.width, canvas.height, 8);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(material);

        sprite.scale.set(canvas.width / 100, canvas.height / 100, 1);
        sprite.position.set(position.x, position.y + 0.5, position.z);

        this.scene.add(sprite);
        this.labelSprites.push(sprite);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    // --- Director Commands ---

    focusTopic(clusterId) {
        const stats = this.clusterStats[clusterId];
        if (stats && stats.centroid) {
            this.flyTo(stats.centroid, 1.5);
        }
    }

    flyTo(target, zoom = 1) {
        const dist = 5 / zoom;
        const startPos = this.camera.position.clone();
        const endPos = new THREE.Vector3(target.x, target.y, target.z + dist);

        // Simple lerp for now (Tween.js would be better)
        this.camera.position.lerp(endPos, 0.1);
        this.controls.target.lerp(new THREE.Vector3(target.x, target.y, target.z), 0.1);
    }

    highlightPoints(ids) {
        const idSet = new Set(ids);
        this.pointMeshes.forEach(mesh => {
            if (idSet.has(mesh.userData.id)) {
                mesh.material.emissiveIntensity = 0.8;
                mesh.scale.set(1.5, 1.5, 1.5);
                // Make bright white/yellow
                mesh.material.color.setHex(0xffffff);
            } else {
                mesh.material.emissiveIntensity = 0.1;
                mesh.scale.set(1, 1, 1);
                mesh.material.color.setHex(0x333333); // Dim others
                mesh.material.transparent = true;
                mesh.material.opacity = 0.2;
            }
        });
    }

    resetView() {
        this.pointMeshes.forEach(mesh => {
            mesh.material.color.copy(mesh.userData.defaultColor);
            mesh.material.emissiveIntensity = 0.4;
            mesh.scale.set(1, 1, 1);
            mesh.material.transparent = false;
            mesh.material.opacity = 1.0;
        });

        if (this.centroid) {
            this.flyTo(this.centroid, 0.8);
        }
    }

    setupInteractions() {
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        window.addEventListener('mousemove', (e) => {
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObjects(this.pointMeshes);

            const tooltip = document.getElementById('tooltip');
            if (intersects.length > 0) {
                const d = intersects[0].object.userData;
                tooltip.style.display = 'block';
                tooltip.style.left = (e.clientX + 10) + 'px';
                tooltip.style.top = (e.clientY + 10) + 'px';
                tooltip.innerHTML = `<strong>${d.score}</strong> | ${d.text.substring(0, 100)}...`;
            } else {
                tooltip.style.display = 'none';
            }
        });
    }

    setColors(model) {
        this.currentModel = model;
        this.pointMeshes.forEach(mesh => {
            const color = this.getColorForPoint(mesh.userData, model);
            mesh.userData.defaultColor = color; // Update default
            mesh.material.color = color;
            mesh.material.emissive = color;
        });
        this.createLabels(); // Re-do labels
    }
}

// Initialize
const stage = new StageController();
window.stage = stage; // Expose for debugging

// UI Logic
document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

async function sendMessage() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    if (!text) return;

    // Add User Message
    addMessage(text, 'user');
    input.value = '';

    // Call API with selected model
    const chatModel = document.getElementById('chat-model').value;
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, model: chatModel })
        });
        const data = await res.json();

        // Add AI Message
        addMessage(data.answer, 'ai');

        // Execute Actions
        if (data.actions) {
            data.actions.forEach(action => {
                console.log("Action:", action);
                if (action.type === 'focus_topic') stage.focusTopic(action.target);
                if (action.type === 'highlight_points') stage.highlightPoints(action.target); 
                if (action.type === 'reset') stage.resetView();
            });
        }

    } catch (e) {
        console.error(e);
        addMessage("Error communicating with Director.", 'ai');
    }
}

function addMessage(text, role) {
    const history = document.getElementById('chat-history');
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
}

// Model Selectors
document.querySelectorAll('#model-selector button').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('#model-selector button').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        stage.setColors(e.target.dataset.model);
    });
});

document.getElementById('reset-btn').addEventListener('click', () => stage.resetView());
