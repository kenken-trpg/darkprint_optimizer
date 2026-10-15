use wasm_bindgen::prelude::*;
use std::arch::wasm32::*;

#[wasm_bindgen]
pub fn process_image(data: &[u8], _width: usize, _height: usize) -> Vec<u8> {

    let mut out = data.to_vec();
    let len = data.len();

    unsafe {
        let mut i = 0;

        while i + 16 <= len {
            // 16バイトロード（RGBA × 4ピクセル）
            let px = v128_load(data.as_ptr().add(i) as *const v128);

            // SIMDで一括処理するための簡易的なロジック
            // 本来はf32に変換してグレースケール計算するのが正確ですが、
            // ここでは一旦、バイト単位での処理例を示します。
            
            let mut pixels = [0u8; 16];
            v128_store(pixels.as_mut_ptr() as *mut v128, px);

            for j in (0..16).step_by(4) {
                let r = pixels[j] as u32;
                let g = pixels[j+1] as u32;
                let b = pixels[j+2] as u32;
                let gray = (r + g + b) / 3;

                if gray < 85 {
                    // 背景を白に
                    pixels[j] = 255;
                    pixels[j+1] = 255;
                    pixels[j+2] = 255;
                } else if gray > 200 {
                    // テキストを黒に
                    pixels[j] = 20;
                    pixels[j+1] = 20;
                    pixels[j+2] = 20;
                } else {
                    // その他も白に（背景化）
                    pixels[j] = 255;
                    pixels[j+1] = 255;
                    pixels[j+2] = 255;
                }
            }

            v128_store(out.as_mut_ptr().add(i) as *mut v128, v128_load(pixels.as_ptr() as *const v128));

            i += 16;
        }
    }

    out
}