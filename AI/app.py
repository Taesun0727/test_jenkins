# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:09:55 2023

@author: SSAFY
"""

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, PNDMScheduler
from diffusers.utils.testing_utils import load_image
import cv2
from PIL import Image
import numpy as np
import torch
from flask import Flask
from io import BytesIO
from flask import send_file

app = Flask(__name__)

def init():
    global pipe
    controlnet_model = "lllyasviel/sd-controlnet-canny"
    sd_model = "Lykon/DreamShaper"
    
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )
    
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

def img2img(img_path, prompt, negative_prompt, num_steps=20, guidance_scale=7, seed=0, low=100, high=200):
    image = load_image(img_path)
    image.thumbnail((512, 512))
    np_image = np.array(image)

    canny_image = cv2.Canny(np_image, low, high)

    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    out_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=torch.manual_seed(seed),
        image=canny_image
    ).images[0]

    return out_image

@app.route('/nfts', methods=["GET"])
def index():
    prompt = "(8k, best quality, masterpiece:1.2),(best quality:1.0), (ultra highres:1.0), a dog, watercolor, by agnes cecile, portrait, extremely luminous bright design, pastel colors, (ink:1.3), autumn lights"
    negative_prompt = "canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"
    num_steps = 20
    guidance_scale = 7
    seed = 3467120481370323442
    
    out_image = img2img("https://people.com/thmb/CjivdYdmbNoUaEblEoFlYdF7qBU=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(584x295:586x297)/coco-rescue-dog-041123-tout-34c967d00e574401aebf75325e38b14e.jpg", prompt, negative_prompt, num_steps, guidance_scale, seed)
    
    # Save the image to a buffer
    buffer = BytesIO()
    out_image.save(buffer, 'JPEG')
    buffer.seek(0)

    return send_file(buffer, mimetype="image/jpeg")

if __name__ == '__main__':
    init()
    app.run(debug=True, port=9999)
    