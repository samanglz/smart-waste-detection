const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

let running = false;
let loopStarted = false;

document.getElementById("startBtn").onclick = startCamera;

async function startCamera() {
    if (running) return;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });

        video.srcObject = stream;
        running = true;

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            if (!loopStarted) {
                loopStarted = true;
                processFrame();
            }
        };

    } catch (error) {
        console.error("Camera error:", error);
    }
}

async function processFrame() {
    if (!running) return;

    try {
        // Capture current camera frame
        const temp = document.createElement("canvas");
        temp.width = video.videoWidth;
        temp.height = video.videoHeight;

        const tctx = temp.getContext("2d");
        tctx.drawImage(
            video,
            0,
            0,
            temp.width,
            temp.height
        );

        const blob = await new Promise(resolve =>
            temp.toBlob(
                resolve,
                "image/jpeg",
                0.85
            )
        );

        if (!blob) {
            throw new Error("Failed to create JPEG frame.");
        }

        const form = new FormData();
        form.append("file", blob, "frame.jpg");

        const start = performance.now();

        const response = await fetch("/predict/live", {
            method: "POST",
            body: form
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        const elapsed = performance.now() - start;

        draw(data.predictions);

        document.getElementById("fps").innerText =
            `API FPS ${(1000 / elapsed).toFixed(1)}`;

    } catch (error) {
        console.error("Live inference error:", error);
    }

    // Wait before sending the next frame
    if (running) {
        setTimeout(processFrame, 300);
    }
}


function draw(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 2;
    ctx.font = "16px Arial";

    predictions.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox;

        ctx.strokeStyle = "#00FF66";
        ctx.strokeRect(
            x1,
            y1,
            x2 - x1,
            y2 - y1
        );

        const label = `${det.class_name} ${det.confidence.toFixed(2)}`;

        ctx.fillStyle = "#00FF66";
        ctx.fillText(
            label,
            x1,
            Math.max(16, y1 - 5)
        );
    });
}