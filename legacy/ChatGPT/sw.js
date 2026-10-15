const CACHE_NAME = "print-optimizer-v1";

const ASSETS = [
    "/",
    "/index.html",
    "/main.js",
    "/worker.js",
    "/style.css",
    "/manifest.json",
    "/pkg/app.js",
    "/pkg/app_bg.wasm"
];

self.addEventListener("install", (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
    );
});

self.addEventListener("fetch", (e) => {
    e.respondWith(
        caches.match(e.request).then(res => res || fetch(e.request))
    );
});