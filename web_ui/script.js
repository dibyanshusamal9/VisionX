const cursor = document.getElementById('gaze-cursor');
const wsStatus = document.getElementById('ws-status');
const buttons = Array.from(document.querySelectorAll('.gaze-btn'));

// UI State
let state = {
    temp: 21,
    vol: 40,
    media: "Bluetooth Audio",
    nav: "No destination set",
    call: "Idle"
};

// UI Elements
const tempDisplay = document.getElementById('temp-display');
const volDisplay = document.getElementById('vol-display');
const mediaDisplay = document.getElementById('media-display');
const navDisplay = document.getElementById('nav-display');
const callDisplay = document.getElementById('call-display');
const toast = document.getElementById('feedback-toast');

let hoveredBtn = null;
const SNAP_RADIUS = 120; // Snap distance in pixels
let toastTimeout = null;

// Initialize Websocket
let ws_conn = null;

function connectWS() {
    const ws = new WebSocket('ws://localhost:8765');
    ws_conn = ws;
    
    ws.onopen = () => {
        wsStatus.classList.remove('disconnected');
        wsStatus.classList.add('connected');
    };
    
    ws.onclose = () => {
        wsStatus.classList.remove('connected');
        wsStatus.classList.add('disconnected');
        setTimeout(connectWS, 2000); // Reconnect
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'cursor') {
            updateCursor(data.x, data.y);
        } else if (data.type === 'click') {
            handleSpaceClick();
        }
    };
}

// Magnetic Snap Logic
function updateCursor(normX, normY) {
    let rawX = normX * window.innerWidth;
    let rawY = normY * window.innerHeight;
    
    let targetX = rawX;
    let targetY = rawY;
    let newHovered = null;
    let minDistance = Infinity;

    // Check all buttons for snapping
    buttons.forEach(btn => {
        const rect = btn.getBoundingClientRect();
        // Calculate center of the button
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const dist = Math.hypot(rawX - centerX, rawY - centerY);
        
        if (dist < SNAP_RADIUS && dist < minDistance) {
            minDistance = dist;
            newHovered = btn;
            // Snap target coordinates to button center
            targetX = centerX;
            targetY = centerY;
        }
    });

    // Update visuals
    if (newHovered !== hoveredBtn) {
        if (hoveredBtn) hoveredBtn.classList.remove('hovered');
        hoveredBtn = newHovered;
        if (hoveredBtn) hoveredBtn.classList.add('hovered');
    }

    if (hoveredBtn) {
        cursor.classList.add('snapped');
    } else {
        cursor.classList.remove('snapped');
    }

    // Move cursor (CSS handles the smooth transition)
    cursor.style.transform = `translate(calc(${targetX}px - 50%), calc(${targetY}px - 50%))`;

    // Sync back to Python
    if (ws_conn && ws_conn.readyState === WebSocket.OPEN) {
        ws_conn.send(JSON.stringify({
            type: "sync_cursor",
            x: targetX / window.innerWidth,
            y: targetY / window.innerHeight
        }));
    }
}

// Actions
function handleSpaceClick() {
    if (hoveredBtn) {
        // Visual feedback
        hoveredBtn.classList.add('active-click');
        setTimeout(() => hoveredBtn.classList.remove('active-click'), 150);
        
        // Execute action
        const action = hoveredBtn.dataset.action;
        executeAction(action);
    }
}

function executeAction(action) {
    let msg = "";
    switch(action) {
        case 'temp_down':
            state.temp = Math.max(16, state.temp - 1);
            tempDisplay.innerHTML = `${state.temp}&deg;C`;
            msg = `Temperature: ${state.temp}°C`;
            break;
        case 'temp_up':
            state.temp = Math.min(28, state.temp + 1);
            tempDisplay.innerHTML = `${state.temp}&deg;C`;
            msg = `Temperature: ${state.temp}°C`;
            break;
        case 'vol_down':
            state.vol = Math.max(0, state.vol - 5);
            volDisplay.innerText = `Vol: ${state.vol}%`;
            msg = `Volume: ${state.vol}%`;
            break;
        case 'vol_up':
            state.vol = Math.min(100, state.vol + 5);
            volDisplay.innerText = `Vol: ${state.vol}%`;
            msg = `Volume: ${state.vol}%`;
            break;
        case 'toggle_media':
            state.media = state.media === "Bluetooth Audio" ? "Radio FM 98.3" : "Bluetooth Audio";
            mediaDisplay.innerText = state.media;
            msg = `Source: ${state.media}`;
            break;
        case 'nav_home':
            state.nav = "Navigating to Home";
            navDisplay.innerText = state.nav;
            msg = "Navigating Home";
            break;
        case 'nav_work':
            state.nav = "Navigating to Work";
            navDisplay.innerText = state.nav;
            msg = "Navigating Work";
            break;
        case 'answer_call':
            state.call = "Call connected";
            callDisplay.innerText = state.call;
            msg = "Call answered";
            break;
    }
    showToast(msg);
}

function showToast(msg) {
    toast.innerText = `> ${msg}`;
    toast.classList.add('show');
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.remove('show');
    }, 2000);
}

// For mouse testing without gaze tracker running
document.addEventListener('mousemove', (e) => {
    // Only use mouse if websocket is disconnected
    if (wsStatus.classList.contains('disconnected')) {
        updateCursor(e.clientX / window.innerWidth, e.clientY / window.innerHeight);
    }
});

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
        e.preventDefault(); // Always prevent spacebar from scrolling the page!
        handleSpaceClick(); // Allow clicking even when websocket is connected
    }
});

connectWS();
