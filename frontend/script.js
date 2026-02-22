/* ═══════════════════════════════════════════════════════════════
   FedFortress — Script
   Three.js 3D Topology · Chart.js · SSE Streaming
   ═══════════════════════════════════════════════════════════════ */

const API = 'http://localhost:5000/api';

// ═══ COLORS ═══
const C = {
    bg: 0x020617,      // Slate 950
    honest: 0x10b981,  // Emerald 500
    malicious: 0xe11d48, // Rose 600
    server: 0x0ea5e9,  // Sky 600
    line: 0x334155,    // Slate 700 (subtle lines)
    ring: 0x1e293b,    // Slate 800
};

// ═══ STATE ═══
let accuracyChart = null;
let lossChart = null;
let scene, camera, renderer, animationId;
let clientMeshes = [];
let lineMeshes = [];
let serverMesh = null;
let orbitAngle = 0;

// ═══════════════════════════════════════════════════════════════
//  3D HERO SCENE — Immersive Neural Network Shield
// ═══════════════════════════════════════════════════════════════
let heroScene, heroCamera, heroRenderer, heroAnimId;
let shieldMesh, heroNodes = [], heroEdges = [];
let mouseTarget = { x: 0, y: 0 };
let mouseCurrent = { x: 0, y: 0 };

function initHero3D() {
    const canvas = document.getElementById('hero-3d');
    if (!canvas) return;
    const w = window.innerWidth;
    const h = window.innerHeight;

    // Scene
    heroScene = new THREE.Scene();
    heroScene.fog = new THREE.FogExp2(0x020617, 0.08);

    // Camera
    heroCamera = new THREE.PerspectiveCamera(55, w / h, 0.1, 100);
    heroCamera.position.set(0, 0, 6);

    // Renderer
    heroRenderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    heroRenderer.setSize(w, h);
    heroRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    heroRenderer.setClearColor(0x020617, 1);

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.15);
    heroScene.add(ambient);
    const p1 = new THREE.PointLight(0x0ea5e9, 2.5, 25);
    p1.position.set(3, 3, 4);
    heroScene.add(p1);
    const p2 = new THREE.PointLight(0x8b5cf6, 1.8, 20);
    p2.position.set(-4, -2, 3);
    heroScene.add(p2);
    const p3 = new THREE.PointLight(0x10b981, 1.2, 18);
    p3.position.set(0, -4, 2);
    heroScene.add(p3);

    // ── Central Shield (Icosahedron wireframe) ──
    const shieldGeo = new THREE.IcosahedronGeometry(1.6, 1);
    const shieldMat = new THREE.MeshPhongMaterial({
        color: 0x0ea5e9,
        emissive: 0x0ea5e9,
        emissiveIntensity: 0.15,
        wireframe: true,
        transparent: true,
        opacity: 0.35,
    });
    shieldMesh = new THREE.Mesh(shieldGeo, shieldMat);
    heroScene.add(shieldMesh);

    // Inner solid core with glow
    const coreGeo = new THREE.IcosahedronGeometry(0.6, 2);
    const coreMat = new THREE.MeshPhongMaterial({
        color: 0x0ea5e9,
        emissive: 0x0ea5e9,
        emissiveIntensity: 0.4,
        transparent: true,
        opacity: 0.2,
    });
    const core = new THREE.Mesh(coreGeo, coreMat);
    shieldMesh.add(core);

    // ── Orbiting data nodes ──
    const nodeCount = 14;
    const nodeColors = [0x10b981, 0x0ea5e9, 0x8b5cf6, 0xe11d48, 0xd97706];
    for (let i = 0; i < nodeCount; i++) {
        const isMal = i >= nodeCount - 3;
        const color = isMal ? 0xe11d48 : nodeColors[i % 3];
        const size = 0.06 + Math.random() * 0.06;
        const geo = new THREE.SphereGeometry(size, 12, 12);
        const mat = new THREE.MeshPhongMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.9,
        });
        const node = new THREE.Mesh(geo, mat);
        // Distribute in a sphere around the shield
        const phi = Math.acos(1 - 2 * (i + 0.5) / nodeCount);
        const theta = Math.PI * (1 + Math.sqrt(5)) * i;
        const r = 2.8 + Math.random() * 0.8;
        node.userData = {
            baseR: r,
            phi,
            theta,
            speed: 0.15 + Math.random() * 0.2,
            phase: Math.random() * Math.PI * 2,
            isMal,
        };
        heroScene.add(node);
        heroNodes.push(node);
    }

    // ── Connection lines between nodes ──
    for (let i = 0; i < nodeCount; i++) {
        for (let j = i + 1; j < nodeCount; j++) {
            if (Math.random() > 0.35) continue; // Sparse connections
            const lineGeo = new THREE.BufferGeometry();
            const positions = new Float32Array(6);
            lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const isMalLine = heroNodes[i].userData.isMal || heroNodes[j].userData.isMal;
            const lineMat = new THREE.LineBasicMaterial({
                color: isMalLine ? 0xe11d48 : 0x0ea5e9,
                transparent: true,
                opacity: isMalLine ? 0.08 : 0.06,
            });
            const line = new THREE.Line(lineGeo, lineMat);
            line.userData = { from: i, to: j };
            heroScene.add(line);
            heroEdges.push(line);
        }
    }

    // ── Background star field ──
    const starGeo = new THREE.BufferGeometry();
    const starCount = 300;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
        starPositions[i * 3] = (Math.random() - 0.5) * 40;
        starPositions[i * 3 + 1] = (Math.random() - 0.5) * 40;
        starPositions[i * 3 + 2] = (Math.random() - 0.5) * 40 - 10;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMat = new THREE.PointsMaterial({ color: 0x94a3b8, size: 0.05, transparent: true, opacity: 0.6 });
    const stars = new THREE.Points(starGeo, starMat);
    heroScene.add(stars);

    // ── Mouse parallax ──
    document.addEventListener('mousemove', (e) => {
        mouseTarget.x = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseTarget.y = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    // ── Animate ──
    let t = 0;
    function animateHero() {
        heroAnimId = requestAnimationFrame(animateHero);
        t += 0.008;

        // Smooth mouse follow
        mouseCurrent.x += (mouseTarget.x - mouseCurrent.x) * 0.03;
        mouseCurrent.y += (mouseTarget.y - mouseCurrent.y) * 0.03;

        // Camera parallax
        heroCamera.position.x = mouseCurrent.x * 0.8;
        heroCamera.position.y = -mouseCurrent.y * 0.5;
        heroCamera.lookAt(0, 0, 0);

        // Shield rotation
        shieldMesh.rotation.x = t * 0.3 + mouseCurrent.y * 0.15;
        shieldMesh.rotation.y = t * 0.5 + mouseCurrent.x * 0.15;

        // Shield pulse
        const pulse = 1 + Math.sin(t * 2) * 0.03;
        shieldMesh.scale.set(pulse, pulse, pulse);

        // Orbit nodes
        heroNodes.forEach((node, i) => {
            const d = node.userData;
            const angle = d.theta + t * d.speed;
            const r = d.baseR + Math.sin(t * 1.5 + d.phase) * 0.3;
            node.position.x = r * Math.sin(d.phi) * Math.cos(angle);
            node.position.y = r * Math.cos(d.phi) + Math.sin(t + d.phase) * 0.2;
            node.position.z = r * Math.sin(d.phi) * Math.sin(angle);
        });

        // Update connection lines
        heroEdges.forEach(line => {
            const { from, to } = line.userData;
            const pos = line.geometry.attributes.position.array;
            const a = heroNodes[from].position;
            const b = heroNodes[to].position;
            pos[0] = a.x; pos[1] = a.y; pos[2] = a.z;
            pos[3] = b.x; pos[4] = b.y; pos[5] = b.z;
            line.geometry.attributes.position.needsUpdate = true;
            // Pulse line opacity
            const dist = a.distanceTo(b);
            line.material.opacity = dist < 5 ? 0.12 * (1 - dist / 6) : 0;
        });

        // Rotate stars slowly
        stars.rotation.y = t * 0.02;
        stars.rotation.x = t * 0.01;

        heroRenderer.render(heroScene, heroCamera);
    }
    animateHero();

    // Resize
    window.addEventListener('resize', () => {
        const w = window.innerWidth;
        const h = window.innerHeight;
        heroCamera.aspect = w / h;
        heroCamera.updateProjectionMatrix();
        heroRenderer.setSize(w, h);
    });
}


// ═══════════════════════════════════════════════════════════════
//  THREE.JS 3D NETWORK TOPOLOGY
// ═══════════════════════════════════════════════════════════════
function init3DTopology() {
    const canvas = document.getElementById('network3d');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(C.bg, 0.15);

    // Camera
    camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(4, 3, 4);
    camera.lookAt(0, 0.5, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(C.bg, 1);

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambient);
    const point = new THREE.PointLight(0x0ea5e9, 1.5, 20);
    point.position.set(0, 4, 0);
    scene.add(point);
    const point2 = new THREE.PointLight(0x8b5cf6, 0.8, 20);
    point2.position.set(-3, 2, 3);
    scene.add(point2);

    // Ground grid
    const gridHelper = new THREE.GridHelper(8, 16, 0x1e293b, 0x1e293b);
    gridHelper.position.y = -0.5;
    scene.add(gridHelper);

    // Build network
    buildNetwork();

    // Mouse orbit
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let cameraAngle = { theta: Math.PI / 4, phi: Math.PI / 6 };
    const cameraRadius = 6;

    canvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        prevMouse = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - prevMouse.x;
        const dy = e.clientY - prevMouse.y;
        cameraAngle.theta -= dx * 0.005;
        cameraAngle.phi = Math.max(0.1, Math.min(Math.PI / 2.2, cameraAngle.phi - dy * 0.005));
        prevMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
    }, { passive: false });

    // Auto-rotation
    function animate() {
        animationId = requestAnimationFrame(animate);
        orbitAngle += 0.003;

        // Orbit clients
        const numClients = getConfig().num_clients;
        const malicious = getConfig().malicious_clients;
        clientMeshes.forEach((mesh, i) => {
            const angle = (2 * Math.PI * i / numClients) + orbitAngle;
            mesh.position.x = 2.2 * Math.cos(angle);
            mesh.position.z = 2.2 * Math.sin(angle);
            mesh.position.y = 0.3 + Math.sin(orbitAngle * 2 + i) * 0.15;
        });

        // Update connection lines
        updateLines();

        // Server pulse
        if (serverMesh) {
            const scale = 1 + Math.sin(orbitAngle * 3) * 0.05;
            serverMesh.scale.set(scale, scale, scale);
        }

        // Camera orbit (auto + drag)
        if (!isDragging) {
            cameraAngle.theta += 0.002;
        }
        camera.position.x = cameraRadius * Math.sin(cameraAngle.phi) * Math.cos(cameraAngle.theta);
        camera.position.y = cameraRadius * Math.cos(cameraAngle.phi);
        camera.position.z = cameraRadius * Math.sin(cameraAngle.phi) * Math.sin(cameraAngle.theta);
        camera.lookAt(0, 0.5, 0);

        renderer.render(scene, camera);
    }
    animate();

    // Resize
    window.addEventListener('resize', () => {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

function buildNetwork() {
    const { num_clients, malicious_clients } = getConfig();

    // Clear old
    clientMeshes.forEach(m => scene.remove(m));
    lineMeshes.forEach(m => scene.remove(m));
    if (serverMesh) scene.remove(serverMesh);
    clientMeshes = [];
    lineMeshes = [];

    // Server (center, elevated)
    const serverGeo = new THREE.OctahedronGeometry(0.35, 1);
    const serverMat = new THREE.MeshPhongMaterial({
        color: C.server,
        emissive: C.server,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.95,
    });
    serverMesh = new THREE.Mesh(serverGeo, serverMat);
    serverMesh.position.set(0, 1.5, 0);
    scene.add(serverMesh);

    // Clients
    for (let i = 0; i < num_clients; i++) {
        const isMal = i < malicious_clients;
        const color = isMal ? C.malicious : C.honest;
        const geo = new THREE.SphereGeometry(0.18, 16, 16);
        const mat = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.25,
            transparent: true,
            opacity: 0.9,
        });
        const mesh = new THREE.Mesh(geo, mat);
        const angle = (2 * Math.PI * i / num_clients);
        mesh.position.set(2.2 * Math.cos(angle), 0.3, 2.2 * Math.sin(angle));
        scene.add(mesh);
        clientMeshes.push(mesh);
    }

    // Orbit ring
    const ringGeo = new THREE.RingGeometry(2.15, 2.25, 64);
    const ringMat = new THREE.MeshBasicMaterial({
        color: C.ring,
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.3;
    scene.add(ring);

    // Connection lines
    for (let i = 0; i < num_clients; i++) {
        const lineGeo = new THREE.BufferGeometry();
        const positions = new Float32Array(6);
        lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const isMal = i < malicious_clients;
        const lineMat = new THREE.LineBasicMaterial({
            color: isMal ? C.malicious : C.line,
            transparent: true,
            opacity: isMal ? 0.3 : 0.2,
        });
        const line = new THREE.Line(lineGeo, lineMat);
        scene.add(line);
        lineMeshes.push(line);
    }
}

function updateLines() {
    clientMeshes.forEach((client, i) => {
        if (lineMeshes[i]) {
            const pos = lineMeshes[i].geometry.attributes.position.array;
            pos[0] = client.position.x;
            pos[1] = client.position.y;
            pos[2] = client.position.z;
            pos[3] = serverMesh.position.x;
            pos[4] = serverMesh.position.y;
            pos[5] = serverMesh.position.z;
            lineMeshes[i].geometry.attributes.position.needsUpdate = true;
        }
    });
}


// ═══════════════════════════════════════════════════════════════
//  CONFIGURATION HELPERS
// ═══════════════════════════════════════════════════════════════
function getConfig() {
    return {
        num_clients: parseInt(document.getElementById('num_clients').value),
        malicious_clients: parseInt(document.getElementById('malicious_clients').value),
        rounds: parseInt(document.getElementById('rounds').value),
        local_epochs: parseInt(document.getElementById('local_epochs').value),
        max_samples: parseInt(document.getElementById('max_samples').value),
        aggregation: document.getElementById('aggregation').value,
        dp_enabled: document.getElementById('dp_enabled').checked,
        dp_epsilon: parseFloat(document.getElementById('dp_epsilon').value),
        attack_type: document.getElementById('attack_type').value,
    };
}

// Aggregation button handler
function setAgg(btn) {
    document.querySelectorAll('.agg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('aggregation').value = btn.dataset.val;
    const descs = {
        'FedAvg': 'Basic weighted averaging — fast but vulnerable to poisoning attacks.',
        'Trimmed Mean': 'Trims extreme values before averaging — more robust against outliers.',
        'Median': 'Takes coordinate-wise median — strongest defense but slower convergence.',
    };
    document.getElementById('agg-desc').textContent = descs[btn.dataset.val] || '';
    updateInfoPanel();
}

function initSliders() {
    // Map slider ID to its value display element ID
    const sliders = [
        { id: 'num_clients', valId: 'v-clients', fmt: v => v },
        { id: 'rounds', valId: 'v-rounds', fmt: v => v },
        { id: 'local_epochs', valId: 'v-epochs', fmt: v => v },
        { id: 'max_samples', valId: 'v-samples', fmt: v => parseInt(v).toLocaleString() },
        { id: 'malicious_clients', valId: 'v-mal', fmt: v => v },
        { id: 'dp_epsilon', valId: 'v-epsilon', fmt: v => parseFloat(v).toFixed(1) },
    ];

    sliders.forEach(({ id, valId, fmt }) => {
        const el = document.getElementById(id);
        const valEl = document.getElementById(valId);
        if (el && valEl) {
            el.addEventListener('input', () => {
                valEl.textContent = fmt(el.value);
                updateInfoPanel();
                if (id === 'num_clients') updateMaliciousMax();
            });
        }
    });

    // Rebuild network when clients or malicious changes
    ['num_clients', 'malicious_clients'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            buildNetwork();
        });
    });

    // DP toggle
    document.getElementById('dp_enabled').addEventListener('change', updateInfoPanel);
    document.getElementById('aggregation').addEventListener('change', updateInfoPanel);
}

function updateMaliciousMax() {
    const n = parseInt(document.getElementById('num_clients').value);
    const malEl = document.getElementById('malicious_clients');
    malEl.max = n - 1;
    if (parseInt(malEl.value) >= n) {
        malEl.value = n - 1;
        document.getElementById('v-mal').textContent = n - 1;
    }
}

function updateInfoPanel() {
    const cfg = getConfig();
    // Update topology sidebar info (use safe access in case elements don't exist yet)
    const setTxt = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setTxt('inf-total', cfg.num_clients);
    setTxt('inf-honest', cfg.num_clients - cfg.malicious_clients);
    setTxt('inf-mal', cfg.malicious_clients);
    setTxt('inf-agg', cfg.aggregation);
    setTxt('inf-dp', cfg.dp_enabled ? `ε=${cfg.dp_epsilon}` : 'Disabled');
    setTxt('inf-rounds', cfg.rounds);
    setTxt('inf-samples', cfg.max_samples.toLocaleString());

    // Update DP defense badge
    const dpBadge = document.getElementById('dp-badge');
    const dpDetail = document.getElementById('dp-detail');
    if (dpBadge && dpDetail) {
        if (cfg.dp_enabled) {
            dpBadge.className = 'def-badge green';
            dpBadge.textContent = '✓ ACTIVE';
            dpDetail.textContent = `Gaussian mechanism (ε=${cfg.dp_epsilon}) with L2 norm clipping`;
        } else {
            dpBadge.className = 'def-badge red';
            dpBadge.textContent = '✗ DISABLED';
            dpDetail.textContent = 'Differential privacy is disabled';
        }
    }
}


// ═══════════════════════════════════════════════════════════════
//  CHART.JS INITIALIZATION
// ═══════════════════════════════════════════════════════════════
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
        legend: { display: false },
    },
    scales: {
        x: {
            title: { display: true, text: 'Round', color: '#94a3b8' },
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
    },
};

function initCharts() {
    const accCtx = document.getElementById('acc-chart').getContext('2d');
    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16,185,129,0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#10b981',
                pointBorderColor: '#fff',
                pointRadius: 5,
                borderWidth: 3,
            }],
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    title: { display: true, text: 'Accuracy (%)', color: '#94a3b8' },
                },
            },
        },
    });

    const lossCtx = document.getElementById('loss-chart')?.getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#e11d48',
                backgroundColor: 'rgba(225,29,72,0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#e11d48',
                pointBorderColor: '#fff',
                pointRadius: 5,
                borderWidth: 3,
            }],
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    title: { display: true, text: 'Loss', color: '#94a3b8' },
                },
            },
        },
    });
}

function resetCharts() {
    if (accuracyChart) {
        accuracyChart.data.labels = [];
        accuracyChart.data.datasets[0].data = [];
        accuracyChart.update();
    }
    if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update();
    }
}

function addChartPoint(round, accuracy, loss) {
    if (accuracyChart) {
        accuracyChart.data.labels.push(`R${round}`);
        accuracyChart.data.datasets[0].data.push(accuracy);
        accuracyChart.update('none');
    }
    if (lossChart) {
        lossChart.data.labels.push(`R${round}`);
        lossChart.data.datasets[0].data.push(loss);
        lossChart.update('none');
    }
}


// ═══════════════════════════════════════════════════════════════
//  THREAT GAUGE
// ═══════════════════════════════════════════════════════════════
function updateGauge(score) {
    score = Math.min(100, Math.max(0, score));
    const arc = document.getElementById('gauge-path');
    const val = document.getElementById('gauge-num');
    const label = document.getElementById('gauge-lbl');
    if (!arc || !val || !label) return;

    // Arc length: full arc is ~267 units (matches SVG path)
    const dashLen = (score / 100) * 267;
    arc.setAttribute('stroke-dasharray', `${dashLen} 267`);
    val.textContent = Math.round(score);

    if (score < 30) label.textContent = 'LOW';
    else if (score < 60) label.textContent = 'MEDIUM';
    else label.textContent = 'HIGH';
}


// ═══════════════════════════════════════════════════════════════
//  SSE TRAINING FUNCTIONS
// ═══════════════════════════════════════════════════════════════
function showSection(id) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('hidden');
}

function setProgress(pct, text) {
    const fill = document.getElementById('prog-fill');
    const label = document.getElementById('prog-label');
    if (fill) fill.style.width = pct + '%';
    if (label) label.textContent = text;
}

function setButtonsDisabled(disabled) {
    document.getElementById('btn-baseline').disabled = disabled;
    document.getElementById('btn-federated').disabled = disabled;
}

// ── Run Baseline ──
function runBaseline() {
    const cfg = getConfig();
    setButtonsDisabled(true);
    showSection('progress-wrap');
    showSection('charts-wrap');
    if (!accuracyChart) initCharts();
    resetCharts();

    setProgress(0, '🏃 Starting centralized baseline training...');

    fetch(`${API}/train/baseline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: 5 }),
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    setButtonsDisabled(false);
                    return;
                }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.done) {
                                setProgress(100, '✅ Baseline training complete!');
                                setButtonsDisabled(false);
                                return;
                            }
                            if (data.error) {
                                setProgress(0, '❌ Error: ' + data.error);
                                setButtonsDisabled(false);
                                return;
                            }
                            const pct = (data.epoch / 5) * 100;
                            setProgress(pct, `📈 Epoch ${data.epoch}/5 — Acc: ${data.accuracy.toFixed(1)}% — Loss: ${data.loss.toFixed(4)}`);
                            addChartPoint(data.epoch, data.accuracy, data.loss);
                        } catch (e) { }
                    }
                });
                read();
            });
        }
        read();
    }).catch(err => {
        setProgress(0, '❌ Connection failed: ' + err.message);
        setButtonsDisabled(false);
    });
}

// ── Run Federated Training ──
function runFederated() {
    const cfg = getConfig();
    setButtonsDisabled(true);
    showSection('progress-wrap');
    showSection('charts-wrap');
    if (!accuracyChart) initCharts();
    resetCharts();

    setProgress(0, '🏃 Initializing federated training on CIFAR-10...');

    const allResults = [];

    fetch(`${API}/train/federated`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    finishTraining(allResults, cfg);
                    setButtonsDisabled(false);
                    return;
                }
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();

                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.done) {
                                finishTraining(allResults, cfg);
                                setButtonsDisabled(false);
                                return;
                            }
                            if (data.error) {
                                setProgress(0, '❌ Error: ' + data.error);
                                setButtonsDisabled(false);
                                return;
                            }
                            allResults.push(data);
                            const totalRounds = cfg.rounds;
                            const pct = (data.round / totalRounds) * 100;
                            setProgress(pct, `🔄 Round ${data.round}/${totalRounds} — Accuracy: ${data.accuracy}% — Loss: ${data.loss.toFixed(3)}`);
                            addChartPoint(data.round, data.accuracy, data.loss);
                        } catch (e) { }
                    }
                });
                read();
            });
        }
        read();
    }).catch(err => {
        setProgress(0, '❌ Connection failed. Make sure Flask server is running (python app.py)');
        setButtonsDisabled(false);
    });
}

function finishTraining(results, cfg) {
    if (results.length === 0) return;

    setProgress(100, '✅ Federated training complete!');

    // Show client cards
    showSection('clients-wrap');
    const grid = document.getElementById('clients-row');
    if (!grid) return;
    grid.innerHTML = '';
    const last = results[results.length - 1];

    for (let i = 0; i < cfg.num_clients; i++) {
        const isMal = i < cfg.malicious_clients;
        const acc = (last.local_accuracies && last.local_accuracies[i]) || 0;
        const anom = (last.anomaly_scores && last.anomaly_scores[i]) || 0;

        const card = document.createElement('div');
        card.className = `client-card ${isMal ? 'malicious' : 'honest'}`;
        card.innerHTML = `
            <div class="client-emoji">${isMal ? '⚠️' : '✅'}</div>
            <div class="client-name">Client ${i}</div>
            <div class="client-type">${isMal ? 'Malicious' : 'Honest'}</div>
            <div class="client-acc" style="color:${isMal ? '#e11d48' : '#10b981'}">${acc.toFixed(1)}%</div>
            <div class="client-anom">anomaly: ${anom.toFixed(1)}</div>
        `;
        grid.appendChild(card);
    }

    // Show security
    showSection('security-section');
    const malRatio = cfg.malicious_clients / cfg.num_clients;
    const threat = malRatio * 70 + Math.random() * 15;
    updateGauge(threat);

    // Show results
    showSection('results');
    const first = results[0];
    const setM = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    setM('m-init', first.accuracy + '%');
    setM('m-final', last.accuracy + '%');
    const improvement = last.accuracy - first.accuracy;
    const deltaEl = document.getElementById('m-delta');
    if (deltaEl) {
        deltaEl.textContent = (improvement >= 0 ? '+' : '') + improvement.toFixed(2) + '%';
        deltaEl.style.color = improvement >= 0 ? '#10b981' : '#e11d48';
    }
    const noise = cfg.dp_enabled ? (0.05 / cfg.dp_epsilon).toFixed(4) : 'N/A';
    setM('m-noise', noise);
    setM('m-mal', `${cfg.malicious_clients}/${cfg.num_clients}`);
    setM('m-rounds', cfg.rounds);

    // Recap table
    const recap = document.getElementById('recap-kv');
    if (recap) {
        recap.innerHTML = `
            <div class="kv-row"><span>Aggregation</span><strong>${cfg.aggregation}</strong></div>
            <div class="kv-row"><span>Clients</span><strong>${cfg.num_clients} (${cfg.malicious_clients} malicious · ${cfg.attack_type.replace(/_/g, ' ')})</strong></div>
            <div class="kv-row"><span>Rounds × Epochs</span><strong>${cfg.rounds} × ${cfg.local_epochs}</strong></div>
            <div class="kv-row"><span>Dataset</span><strong>CIFAR-10 (${cfg.max_samples.toLocaleString()} samples)</strong></div>
            <div class="kv-row"><span>Differential Privacy</span><strong>${cfg.dp_enabled ? 'ε=' + cfg.dp_epsilon : 'Disabled'}</strong></div>
        `;
    }
}


// ═══════════════════════════════════════════════════════════════
//  STATIC DEMO CHARTS
// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════
//  STATIC DEMO CHARTS — RICH ANALYTICS
// ═══════════════════════════════════════════════════════════════
function initDemoCharts() {
    const theme = {
        grid: 'rgba(255,255,255,0.03)',
        text: '#94a3b8', // Slate 400
        blue: '#0ea5e9', // Sapphire
        green: '#10b981', // Emerald
        red: '#e11d48',   // Rose
        purple: '#8b5cf6', // Violet
        amber: '#d97706',  // Amber
    };

    const commonOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1000, easing: 'easeOutQuart' },
        plugins: {
            legend: {
                display: true,
                position: 'bottom',
                labels: { color: theme.text, boxWidth: 12, padding: 15, font: { size: 11 } }
            },
            tooltip: {
                backgroundColor: '#1e293b',
                titleColor: '#fff',
                bodyColor: '#cbd5e1',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 10
            }
        },
        scales: {
            x: {
                ticks: { color: theme.text, font: { size: 10 } },
                grid: { color: theme.grid },
            },
            y: {
                ticks: { color: theme.text, font: { size: 10 } },
                grid: { color: theme.grid },
            },
        },
    };

    // 1. Multi-Series Accuracy Chart
    const accCtx = document.getElementById('demo-acc-chart')?.getContext('2d');
    if (accCtx) {
        new Chart(accCtx, {
            type: 'line',
            data: {
                labels: ['R1', 'R2', 'R3', 'R4', 'R5'],
                datasets: [
                    {
                        label: 'Global Model',
                        data: [32.4, 41.2, 48.7, 53.1, 58.9],
                        borderColor: theme.blue,
                        backgroundColor: 'rgba(14,165,233,0.1)',
                        fill: true,
                        tension: 0.4,
                        borderWidth: 3,
                        pointRadius: 4,
                    },
                    {
                        label: 'Honest Avg',
                        data: [34.1, 44.5, 51.2, 56.8, 60.5],
                        borderColor: theme.green,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                    },
                    {
                        label: 'Malicious impact',
                        data: [28.5, 25.1, 21.3, 18.4, 17.0],
                        borderColor: theme.red,
                        borderDash: [2, 2],
                        fill: false,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 0,
                    }
                ]
            },
            options: {
                ...commonOpts,
                scales: {
                    ...commonOpts.scales,
                    y: { ...commonOpts.scales.y, min: 0, max: 80, title: { display: true, text: 'Accuracy %', color: theme.text } }
                }
            }
        });
    }

    // 2. Training Loss Chart
    const lossCtx = document.getElementById('demo-loss-chart')?.getContext('2d');
    if (lossCtx) {
        new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: ['R1', 'R2', 'R3', 'R4', 'R5'],
                datasets: [{
                    label: 'Cross-Entropy Loss',
                    data: [2.15, 1.78, 1.42, 1.15, 0.94],
                    borderColor: theme.red,
                    backgroundColor: 'rgba(225,29,72,0.1)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2,
                }]
            },
            options: {
                ...commonOpts,
                plugins: { ...commonOpts.plugins, legend: { display: false } },
                scales: {
                    ...commonOpts.scales,
                    y: { ...commonOpts.scales.y, min: 0, max: 2.5 }
                }
            }
        });
    }

    // 3. Per-Client Accuracy (Final Round) — BAR CHART
    const clientCtx = document.getElementById('demo-client-bar')?.getContext('2d');
    if (clientCtx) {
        new Chart(clientCtx, {
            type: 'bar',
            data: {
                labels: ['C0', 'C1', 'C2', 'C3 (M)', 'C4 (M)'],
                datasets: [{
                    label: 'Local Accuracy %',
                    data: [61.2, 59.8, 60.5, 18.7, 15.3],
                    backgroundColor: [theme.green, theme.green, theme.green, theme.red, theme.red],
                    borderRadius: 4,
                }]
            },
            options: {
                ...commonOpts,
                plugins: { ...commonOpts.plugins, legend: { display: false } },
                scales: {
                    ...commonOpts.scales,
                    y: { ...commonOpts.scales.y, min: 0, max: 100 }
                }
            }
        });
    }

    // 4. Anomaly score per client — BAR CHART
    const anomCtx = document.getElementById('demo-anomaly-bar')?.getContext('2d');
    if (anomCtx) {
        new Chart(anomCtx, {
            type: 'bar',
            data: {
                labels: ['C0', 'C1', 'C2', 'C3', 'C4'],
                datasets: [{
                    label: 'Anomaly Score (AWTM Detection)',
                    data: [0.3, 0.2, 0.4, 8.4, 9.1],
                    backgroundColor: ['rgba(148,163,184,0.3)', 'rgba(148,163,184,0.3)', 'rgba(148,163,184,0.3)', theme.amber, theme.amber],
                    borderRadius: 4,
                }]
            },
            options: {
                ...commonOpts,
                plugins: { ...commonOpts.plugins, legend: { display: false } },
                scales: {
                    ...commonOpts.scales,
                    y: { ...commonOpts.scales.y, min: 0, max: 12, title: { display: true, text: 'Score', color: theme.text } }
                }
            }
        });
    }
}


// ═══════════════════════════════════════════════════════════════
//  SCROLL REVEAL OBSERVER
// ═══════════════════════════════════════════════════════════════
function initScrollReveal() {
    const revealEls = document.querySelectorAll('[data-reveal]');
    const staggerEls = document.querySelectorAll('[data-reveal-stagger]');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

    revealEls.forEach(el => observer.observe(el));

    // Stagger observer — reveals children one by one
    const staggerObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                const children = entry.target.children;
                Array.from(children).forEach((child, i) => {
                    child.style.transitionDelay = `${i * 100}ms`;
                });
                staggerObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -30px 0px' });

    staggerEls.forEach(el => staggerObserver.observe(el));
}

// ═══════════════════════════════════════════════════════════════
//  ACTIVE NAV LINK HIGHLIGHTING
// ═══════════════════════════════════════════════════════════════
function initActiveNav() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-links a');

    const navObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                navLinks.forEach(link => {
                    link.classList.toggle('active',
                        link.getAttribute('href') === `#${entry.target.id}`
                    );
                });
            }
        });
    }, { threshold: 0.3, rootMargin: '-80px 0px -40% 0px' });

    sections.forEach(s => navObserver.observe(s));
}


// ═══════════════════════════════════════════════════════════════
//  API HEALTH CHECK
// ═══════════════════════════════════════════════════════════════
function checkAPIStatus() {
    const dot = document.querySelector('.status-dot');
    const txt = document.getElementById('api-status-text');
    if (!dot || !txt) return;

    fetch(`${API}/status`, { method: 'GET' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ok') {
                dot.classList.add('connected');
                dot.classList.remove('disconnected');
                txt.textContent = 'Connected';
            } else {
                dot.classList.remove('connected');
                dot.classList.add('disconnected');
                txt.textContent = 'Error';
            }
        })
        .catch(() => {
            dot.classList.remove('connected');
            dot.classList.add('disconnected');
            txt.textContent = 'Disconnected';
        });
}


// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initHero3D();
    init3DTopology();
    initSliders();
    updateInfoPanel();
    initDemoCharts();
    initScrollReveal();
    initActiveNav();

    // API health check on load + poll every 10s
    checkAPIStatus();
    setInterval(checkAPIStatus, 10000);

    // Chatbot keyboard handling
    const chatInput = document.getElementById('chatbot-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
});


// ═══════════════════════════════════════════════════════════════
//  MEDICAL AI CHATBOT — Interaction Logic
// ═══════════════════════════════════════════════════════════════

function openChatbot() {
    const overlay = document.getElementById('chatbot-overlay');
    if (!overlay) return;
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
    // Focus input after modal animation
    setTimeout(() => {
        document.getElementById('chatbot-input')?.focus();
    }, 500);
}

function closeChatbot() {
    const overlay = document.getElementById('chatbot-overlay');
    if (!overlay) return;
    overlay.classList.remove('active');
    document.body.style.overflow = '';
}

// Close chatbot on clicking overlay background (not the modal itself)
document.addEventListener('click', (e) => {
    if (e.target.id === 'chatbot-overlay') {
        closeChatbot();
    }
});

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeChatbot();
});

// ── Send message (from input) ──
function sendMessage() {
    const input = document.getElementById('chatbot-input');
    if (!input) return;
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    addMessage(text, 'user');
    processUserMessage(text);
}

// ── Send quick message (from quick action buttons) ──
function sendQuickMsg(text) {
    addMessage(text, 'user');
    // Hide quick actions after first use
    const quickActions = document.querySelector('.chat-quick-actions');
    if (quickActions) {
        quickActions.style.transition = 'opacity 0.3s, transform 0.3s';
        quickActions.style.opacity = '0';
        quickActions.style.transform = 'translateY(-8px)';
        setTimeout(() => quickActions.remove(), 300);
    }
    processUserMessage(text);
}

// ── Add a message to the chat ──
function addMessage(text, sender) {
    const container = document.getElementById('chatbot-messages');
    if (!container) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg chat-${sender}`;

    const avatarEmoji = sender === 'user' ? '👤' : '🧬';

    msgDiv.innerHTML = `
        <div class="chat-msg-avatar">${avatarEmoji}</div>
        <div class="chat-msg-content">
            <div class="chat-msg-bubble"><p>${escapeHTML(text)}</p></div>
            <div class="chat-msg-time">${timeStr}</div>
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

// ── Add bot message with HTML content ──
function addBotMessage(html) {
    const container = document.getElementById('chatbot-messages');
    if (!container) return;

    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg chat-bot';

    msgDiv.innerHTML = `
        <div class="chat-msg-avatar">🧬</div>
        <div class="chat-msg-content">
            <div class="chat-msg-bubble">${html}</div>
            <div class="chat-msg-time">${timeStr}</div>
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Typing indicator ──
function showTyping() {
    const typing = document.getElementById('chat-typing');
    if (typing) typing.classList.add('active');
    // Also scroll to bottom
    const container = document.getElementById('chatbot-messages');
    if (container) container.scrollTop = container.scrollHeight;
}

function hideTyping() {
    const typing = document.getElementById('chat-typing');
    if (typing) typing.classList.remove('active');
}

// ── Process user message ──
// This function currently uses placeholder responses.
// Replace the body of this function with your actual backend call.
function processUserMessage(text) {
    showTyping();

    // ── 🔌 BACKEND HOOK POINT ──
    // Replace the setTimeout below with your actual backend call:
    //
    // fetch('/api/chatbot', {
    //     method: 'POST',
    //     headers: { 'Content-Type': 'application/json' },
    //     body: JSON.stringify({ message: text })
    // })
    // .then(res => res.json())
    // .then(data => {
    //     hideTyping();
    //     addBotMessage(data.response);
    // })
    // .catch(err => {
    //     hideTyping();
    //     addBotMessage('<p>Sorry, I encountered an error. Please try again.</p>');
    // });

    const delay = 1200 + Math.random() * 800;
    setTimeout(() => {
        hideTyping();
        const response = getPlaceholderResponse(text);
        addBotMessage(response);
    }, delay);
}

// ── Placeholder responses (remove when backend is connected) ──
function getPlaceholderResponse(text) {
    const lower = text.toLowerCase();

    if (lower.includes('headache') || lower.includes('fever') || lower.includes('symptom')) {
        return `
            <p><strong>🩺 Symptom Analysis</strong></p>
            <p>Based on your symptoms, here's a preliminary assessment using our federated learning model trained across 50+ hospitals:</p>
            <ul>
                <li><strong>Likely conditions:</strong> Common cold, Tension headache, Viral infection</li>
                <li><strong>Severity:</strong> <span style="color: var(--amber-l)">Mild to Moderate</span></li>
                <li><strong>Recommended:</strong> Rest, hydration, and over-the-counter pain relief</li>
            </ul>
            <p>📊 <em>Confidence: 87% (trained on 2.3M anonymized patient records)</em></p>
            <p class="chat-disclaimer">⚠️ This is an AI-powered assessment. Please consult a healthcare professional for accurate diagnosis.</p>
        `;
    }

    if (lower.includes('ibuprofen') || lower.includes('drug') || lower.includes('medicine') || lower.includes('medication') || lower.includes('side effect')) {
        return `
            <p><strong>💊 Medication Information: Ibuprofen</strong></p>
            <p>Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID).</p>
            <ul>
                <li><strong>Uses:</strong> Pain relief, fever reduction, anti-inflammatory</li>
                <li><strong>Common side effects:</strong> Nausea, stomach upset, dizziness</li>
                <li><strong>Serious risks:</strong> GI bleeding, kidney issues (prolonged use)</li>
                <li><strong>Interactions:</strong> Aspirin, blood thinners, ACE inhibitors</li>
            </ul>
            <p>🔒 <em>Data sourced from federated drug interaction databases across partner hospitals</em></p>
        `;
    }

    if (lower.includes('blood test') || lower.includes('lab') || lower.includes('result')) {
        return `
            <p><strong>🧪 Lab Result Interpretation</strong></p>
            <p>I can help interpret common blood test results. Please share your values, or I can explain typical ranges:</p>
            <ul>
                <li><strong>CBC:</strong> Hemoglobin, WBC count, Platelet count</li>
                <li><strong>Metabolic:</strong> Blood glucose, Creatinine, BUN</li>
                <li><strong>Lipid Panel:</strong> LDL, HDL, Triglycerides</li>
                <li><strong>Liver Function:</strong> ALT, AST, Bilirubin</li>
            </ul>
            <p>📤 <em>Share your lab values and I'll provide a personalized analysis</em></p>
        `;
    }

    if (lower.includes('health tip') || lower.includes('preventive') || lower.includes('wellness')) {
        return `
            <p><strong>🏃 Preventive Health Tips</strong></p>
            <p>Based on federated health data analytics from our privacy-preserving network:</p>
            <ul>
                <li>🥗 <strong>Nutrition:</strong> Aim for 5+ servings of fruits and vegetables daily</li>
                <li>🏋️ <strong>Exercise:</strong> 150 min moderate activity per week</li>
                <li>😴 <strong>Sleep:</strong> Maintain 7-9 hours of quality sleep</li>
                <li>🧘 <strong>Stress:</strong> Practice mindfulness or deep breathing daily</li>
                <li>💧 <strong>Hydration:</strong> Drink 2-3 liters of water per day</li>
            </ul>
            <p>🔐 <em>Personalized insights derived without exposing individual patient data</em></p>
        `;
    }

    // Default response
    return `
        <p>Thank you for your question. Our federated AI model is processing your query across our secure medical knowledge network.</p>
        <p>I can help you with:</p>
        <ul>
            <li>🩺 Symptom analysis</li>
            <li>💊 Medication queries</li>
            <li>🧪 Lab result interpretation</li>
            <li>🏃 Health & wellness tips</li>
        </ul>
        <p>Could you provide more details about your health concern?</p>
    `;
}
