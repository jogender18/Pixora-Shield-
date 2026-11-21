const API_URL = "http://localhost:5000";

const videoInput = document.querySelector("#videoInput");
const dropzone = document.querySelector("#dropzone");
const fileName = document.querySelector("#fileName");
const fileSize = document.querySelector("#fileSize");
const processButton = document.querySelector("#processButton");
const error = document.querySelector("#error");
const privacyCard = document.querySelector("#privacyCard");
const results = document.querySelector("#results");
const beforeVideo = document.querySelector("#beforeVideo");
const afterVideo = document.querySelector("#afterVideo");
const downloadLink = document.querySelector("#downloadLink");
const metrics = document.querySelector("#metrics");
const plateNote = document.querySelector("#plateNote");

let selectedFile = null;

function selectFile(file) {
  if (!file) return;

  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  processButton.disabled = false;
  processButton.textContent = "Blur & protect video";
  error.hidden = true;
  results.hidden = true;
  privacyCard.hidden = false;
  beforeVideo.src = URL.createObjectURL(file);
}

videoInput.addEventListener("change", (event) => {
  selectFile(event.target.files[0]);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragging");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragging");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  selectFile(event.dataTransfer.files[0]);
});

processButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  processButton.disabled = true;
  processButton.innerHTML = '<span class="spinner"></span> Processing video';
  error.hidden = true;
  results.hidden = true;

  const form = new FormData();
  form.append("video", selectedFile);

  try {
    const response = await fetch(`${API_URL}/api/process`, {
      method: "POST",
      body: form,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Processing failed");
    }

    const processedUrl = `${API_URL}${data.processed_url}`;
    afterVideo.src = processedUrl;
    downloadLink.href = `${API_URL}/api/download/${data.processed_url.split("/").pop()}`;

    metrics.innerHTML = [
      [data.metrics.frames, "Frames processed"],
      [data.metrics.faces_and_plates_detected, "Privacy regions found"],
      [`${data.metrics.average_fps} fps`, "Average processing"],
      [data.metrics.resolution, "Resolution"],
    ].map(([value, label]) =>
      `<div><strong>${value}</strong><span>${label}</span></div>`
    ).join("");

    plateNote.hidden = data.metrics.plate_detection_enabled;
    privacyCard.hidden = true;
    results.hidden = false;
  } catch (caught) {
    error.textContent = caught instanceof Error ? caught.message : "Processing failed";
    error.hidden = false;
  } finally {
    processButton.disabled = false;
    processButton.textContent = "Blur & protect video";
  }
});
