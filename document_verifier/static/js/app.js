const startButton = document.getElementById("startCamera");
const captureButton = document.getElementById("capturePhoto");
const video = document.getElementById("camera");
const snapshot = document.getElementById("snapshot");
const cameraGuide = document.getElementById("cameraGuide");
const cameraStatus = document.getElementById("cameraStatus");

if (startButton && captureButton && video && snapshot) {
  let stream = null;
  let guideTimer = null;
  let latestCrop = null;

  const detectDocumentCrop = (canvas, width, height) => {
    const sampleWidth = 240;
    const sampleHeight = Math.max(1, Math.round((height / width) * sampleWidth));
    const sample = document.createElement("canvas");
    sample.width = sampleWidth;
    sample.height = sampleHeight;
    const ctx = sample.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(canvas, 0, 0, sampleWidth, sampleHeight);
    const data = ctx.getImageData(0, 0, sampleWidth, sampleHeight).data;
    let minX = sampleWidth;
    let minY = sampleHeight;
    let maxX = 0;
    let maxY = 0;
    let hits = 0;

    for (let y = 1; y < sampleHeight - 1; y += 1) {
      for (let x = 1; x < sampleWidth - 1; x += 1) {
        const index = (y * sampleWidth + x) * 4;
        const left = (y * sampleWidth + x - 1) * 4;
        const up = ((y - 1) * sampleWidth + x) * 4;
        const brightness = (data[index] + data[index + 1] + data[index + 2]) / 3;
        const neighbor = (data[left] + data[left + 1] + data[left + 2] + data[up] + data[up + 1] + data[up + 2]) / 6;
        if (brightness > 105 && Math.abs(brightness - neighbor) > 18) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
          hits += 1;
        }
      }
    }

    if (hits < 80 || maxX - minX < sampleWidth * 0.35 || maxY - minY < sampleHeight * 0.35) {
      return {
        x: Math.round(width * 0.08),
        y: Math.round(height * 0.08),
        width: Math.round(width * 0.84),
        height: Math.round(height * 0.84),
        detected: false
      };
    }

    const scaleX = width / sampleWidth;
    const scaleY = height / sampleHeight;
    return {
      x: Math.max(0, Math.round(minX * scaleX)),
      y: Math.max(0, Math.round(minY * scaleY)),
      width: Math.min(width, Math.round((maxX - minX) * scaleX)),
      height: Math.min(height, Math.round((maxY - minY) * scaleY)),
      detected: true
    };
  };

  const drawCameraGuide = () => {
    if (!cameraGuide || !video.videoWidth || !video.videoHeight) {
      return;
    }
    const rect = video.getBoundingClientRect();
    cameraGuide.width = rect.width;
    cameraGuide.height = rect.height;
    const frame = document.createElement("canvas");
    frame.width = video.videoWidth;
    frame.height = video.videoHeight;
    frame.getContext("2d").drawImage(video, 0, 0, frame.width, frame.height);
    latestCrop = detectDocumentCrop(frame, frame.width, frame.height);

    const scaleX = rect.width / frame.width;
    const scaleY = rect.height / frame.height;
    const ctx = cameraGuide.getContext("2d");
    ctx.clearRect(0, 0, cameraGuide.width, cameraGuide.height);
    ctx.strokeStyle = "#21d07a";
    ctx.lineWidth = 3;
    ctx.fillStyle = "rgba(33, 208, 122, 0.08)";
    ctx.setLineDash(latestCrop.detected ? [] : [10, 8]);
    ctx.fillRect(latestCrop.x * scaleX, latestCrop.y * scaleY, latestCrop.width * scaleX, latestCrop.height * scaleY);
    ctx.strokeRect(latestCrop.x * scaleX, latestCrop.y * scaleY, latestCrop.width * scaleX, latestCrop.height * scaleY);
    ctx.setLineDash([]);
  };

  const startGuideLoop = () => {
    if (!cameraGuide) {
      return;
    }
    if (guideTimer) {
      clearInterval(guideTimer);
    }
    guideTimer = setInterval(drawCameraGuide, 450);
    drawCameraGuide();
  };

  const stopExistingStream = () => {
    if (!stream) {
      return;
    }
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    if (guideTimer) {
      clearInterval(guideTimer);
      guideTimer = null;
    }
  };

  const setCameraReadyState = (active) => {
    captureButton.disabled = !active;
    startButton.textContent = active ? "Restart Camera" : "Start Camera";
  };

  const startWithConstraints = async (constraints) => {
    stopExistingStream();
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.muted = true;
    video.autoplay = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    await new Promise((resolve) => {
      if (video.readyState >= 2) {
        resolve();
        return;
      }
      video.onloadedmetadata = () => resolve();
    });
    const track = stream.getVideoTracks()[0];
    const settings = track ? track.getSettings() : {};
    const width = settings.width || video.videoWidth;
    const height = settings.height || video.videoHeight;
    cameraStatus.textContent = `Camera ready (${width || "?"}x${height || "?"}).`;
    setCameraReadyState(true);
    startGuideLoop();
  };

  startButton.addEventListener("click", async () => {
    try {
      cameraStatus.textContent = "Starting camera...";
      await startWithConstraints({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      if (!video.videoWidth || !video.videoHeight) {
        cameraStatus.textContent = "Camera connected, but preview is not rendering yet. Retrying...";
        await startWithConstraints({
          video: true,
          audio: false
        });
      }
    } catch (error) {
      setCameraReadyState(false);
      cameraStatus.textContent = `Camera error: ${error.name || "Error"} - ${error.message}`;
    }
  });

  captureButton.addEventListener("click", async () => {
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const sourceCanvas = document.createElement("canvas");
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    const sourceContext = sourceCanvas.getContext("2d");
    sourceContext.drawImage(video, 0, 0, width, height);

    const crop = latestCrop || detectDocumentCrop(sourceCanvas, width, height);
    const margin = 0.035;
    const x = Math.max(0, Math.floor(crop.x - crop.width * margin));
    const y = Math.max(0, Math.floor(crop.y - crop.height * margin));
    const cropWidth = Math.min(width - x, Math.ceil(crop.width * (1 + margin * 2)));
    const cropHeight = Math.min(height - y, Math.ceil(crop.height * (1 + margin * 2)));

    snapshot.width = cropWidth;
    snapshot.height = cropHeight;
    const context = snapshot.getContext("2d");
    context.drawImage(sourceCanvas, x, y, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
    const image = snapshot.toDataURL("image/jpeg", 0.92);
    captureButton.disabled = true;
    cameraStatus.textContent = "Cropping document and verifying scan...";

    const response = await fetch("/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image })
    });
    const payload = await response.json();
    if (!response.ok) {
      cameraStatus.textContent = payload.error || "Capture verification failed.";
      captureButton.disabled = false;
      return;
    }
    window.location.href = `/documents/${payload.document_id}`;
  });

  window.addEventListener("beforeunload", stopExistingStream);
}

const image = document.getElementById("documentImage");
const overlay = document.getElementById("overlay");

if (image && overlay && Array.isArray(window.verificationIssues)) {
  const drawOverlay = () => {
    const rect = image.getBoundingClientRect();
    overlay.width = rect.width;
    overlay.height = rect.height;
    overlay.style.width = `${rect.width}px`;
    overlay.style.height = `${rect.height}px`;
    const ctx = overlay.getContext("2d");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const scaleX = rect.width / image.naturalWidth;
    const scaleY = rect.height / image.naturalHeight;

    const colors = {
      edge_inconsistency: "#ff5400",
      noise_variance_anomaly: "#e0a000",
      localized_blur_anomaly: "#cc2f46"
    };

    window.verificationIssues.forEach((issue) => {
      const box = issue.box;
      const x = box.x * scaleX;
      const y = box.y * scaleY;
      const w = box.width * scaleX;
      const h = box.height * scaleY;
      const color = colors[issue.issue_type] || "#00a35a";
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.fillStyle = `${color}28`;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    });
  };

  image.addEventListener("load", drawOverlay);
  window.addEventListener("resize", drawOverlay);
  if (image.complete) {
    drawOverlay();
  }
}
