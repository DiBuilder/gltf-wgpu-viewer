const mat4 = window.mat4 || (window.glMatrix && window.glMatrix.mat4);
const vec3 = window.vec3 || (window.glMatrix && window.glMatrix.vec3);

export class OrbitCamera {
    constructor(canvas, initialTarget = [0, 0, 0], initialDistance = 10) {
        this.canvas = canvas;
        this.target = vec3.clone(initialTarget);
        this.distance = initialDistance;
        this.minDistance = 0.1;
        this.maxDistance = 1000;
        this.yaw = 0;
        this.pitch = 0;
        
        // State
        this.isDragging = false;
        this.isPanning = false;
        this.lastX = 0;
        this.lastY = 0;
        
        // Configuration
        this.rotateSpeed = 0.005;
        this.panSpeed = 0.002;
        this.zoomSpeed = 0.1;
        
        if (!mat4 || !vec3) {
            console.error("gl-matrix not found! Camera will not work.");
        }

        this.bindEvents();
    }

    bindEvents() {
        this.canvas.addEventListener('contextmenu', e => e.preventDefault());
        
        this.canvas.addEventListener('pointerdown', (e) => {
            this.canvas.setPointerCapture(e.pointerId);
            this.lastX = e.clientX;
            this.lastY = e.clientY;
            
            // Left click + Shift OR Right click OR Middle click -> PAN
            if ((e.button === 0 && e.shiftKey) || e.button === 1 || e.button === 2) {
                this.isPanning = true;
                this.isDragging = false;
                console.log("Camera: Panning started");
            } 
            // Left click -> ROTATE
            else if (e.button === 0) { 
                this.isDragging = true;
                this.isPanning = false;
                console.log("Camera: Rotate started");
            }
        });

        this.canvas.addEventListener('pointermove', (e) => {
            if (!this.isDragging && !this.isPanning) return;
            
            const dx = e.clientX - this.lastX;
            const dy = e.clientY - this.lastY;
            this.lastX = e.clientX;
            this.lastY = e.clientY;

            if (this.isDragging) {
                this.yaw -= dx * this.rotateSpeed;
                this.pitch -= dy * this.rotateSpeed;
                // Clamp pitch to avoid gimbal lock
                const limit = Math.PI * 0.49;
                this.pitch = Math.max(-limit, Math.min(limit, this.pitch));
            }

            if (this.isPanning) {
                this.pan(dx, dy);
            }
        });

        this.canvas.addEventListener('pointerup', (e) => {
            this.isDragging = false;
            this.isPanning = false;
            this.canvas.releasePointerCapture(e.pointerId);
        });
        
        this.canvas.addEventListener('pointerleave', () => {
            this.isDragging = false;
            this.isPanning = false;
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = Math.sign(e.deltaY);
            this.distance *= delta > 0 ? (1 + this.zoomSpeed) : (1 - this.zoomSpeed);
            this.distance = Math.max(this.minDistance, Math.min(this.maxDistance, this.distance));
        }, { passive: false });
    }

    pan(dx, dy) {
        // Calculate camera basis
        const view = this.getViewMatrix();
        // Right vector is row 0 of view matrix (in gl-matrix)
        const right = [view[0], view[4], view[8]];
        // Up vector is row 1
        const up = [view[1], view[5], view[9]];
        
        const speed = this.distance * this.panSpeed;
        
        // Move target opposite to mouse drag
        vec3.scaleAndAdd(this.target, this.target, right, -dx * speed);
        vec3.scaleAndAdd(this.target, this.target, up, dy * speed);
    }

    reset(target, distance) {
        vec3.copy(this.target, target);
        this.distance = distance;
        this.yaw = 0;
        this.pitch = 0;
    }

    getEyePosition() {
        const cosPitch = Math.cos(this.pitch);
        const sinPitch = Math.sin(this.pitch);
        const cosYaw = Math.cos(this.yaw);
        const sinYaw = Math.sin(this.yaw);
        
        return [
            this.target[0] + this.distance * cosPitch * sinYaw,
            this.target[1] + this.distance * sinPitch,
            this.target[2] + this.distance * cosPitch * cosYaw
        ];
    }

    getViewMatrix() {
        const eye = this.getEyePosition();
        const view = mat4.create();
        mat4.lookAt(view, eye, this.target, [0, 1, 0]);
        return view;
    }
}
