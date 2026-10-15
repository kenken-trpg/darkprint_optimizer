const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const worker = new Worker("worker.js");

let currentImageData = null;

worker.onmessage = (e) => {
    currentImageData.data.set(e.data);
    ctx.putImageData(currentImageData, 0, 0);
};

document.getElementById("drop").ondrop = (e) => {
    e.preventDefault();

    const file = e.dataTransfer.files[0];
    const img = new Image();
    img.src = URL.createObjectURL(file);

    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0, 0);

        currentImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        worker.postMessage({
            data: currentImageData.data,
            width: canvas.width,
            height: canvas.height
        });
    };
};

document.getElementById("drop").ondragover = (e) => e.preventDefault();

function download() {
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = "optimized.png";
    a.click();
}

// Service Worker登録
if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js");
}