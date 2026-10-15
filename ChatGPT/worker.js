import init, { process_image } from "./pkg/app.js";

let ready = init();

onmessage = async (e) => {
    await ready;

    const { data, width, height } = e.data;

    const result = process_image(data, width, height);

    postMessage(result);
};