import modules.scripts as scripts
import gradio as gr

from modules import processing, images, shared, sd_samplers
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state, Options

import torch
import cv2
import requests
import os.path

from PIL import ImageChops, Image, ImageOps
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy

from ldm_seg.util import instantiate_from_config
from omegaconf import OmegaConf
import math
import numpy as np
from einops import repeat
import PIL

debug = False

class Script(scripts.Script):
	def title(self):
		return "LD-ZNet txt2mask"

	def show(self, is_img2img):
		return is_img2img

	def ui(self, is_img2img):
		if not is_img2img:
			return None

		mask_prompt = gr.Textbox(label="Mask prompt", lines=1)
		negative_mask_prompt = gr.Textbox(label="Negative mask prompt", lines=1)
		mask_precision = gr.Slider(label="Mask precision", minimum=0.0, maximum=255.0, step=1.0, value=100.0)
		mask_padding = gr.Slider(label="Mask padding", minimum=0.0, maximum=500.0, step=1.0, value=0.0)
		brush_mask_mode = gr.Radio(label="Brush mask mode", choices=['discard','add','subtract'], value='discard', type="index", visible=False)
		mask_output = gr.Checkbox(label="Show mask in output?",value=True)

		plug = gr.HTML(label="plug",value='<div class="gr-block gr-box relative w-full overflow-hidden border-solid border border-gray-200 gr-panel"><p>If you like my work, please consider showing your support on <strong><a href="https://patreon.com/thereforegames" target="_blank">Patreon</a></strong>. Thank you! &#10084;</p></div>')

		return [mask_prompt,negative_mask_prompt, mask_precision, mask_padding, brush_mask_mode, mask_output, plug]

	def run(self, p, mask_prompt, negative_mask_prompt, mask_precision, mask_padding, brush_mask_mode, mask_output, plug):
		def download_file(filename, url):
			with open(filename, 'wb') as fout:
				response = requests.get(url, stream=True)
				response.raise_for_status()
				# Write response data to file
				for block in response.iter_content(4096):
					fout.write(block)
		def pil_to_cv2(img):
			return (cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR))
		def gray_to_pil(img):
			return (Image.fromarray(cv2.cvtColor(img,cv2.COLOR_GRAY2RGBA)))
		
		def center_crop(img,new_width,new_height):
			width, height = img.size   # Get dimensions

			left = (width - new_width)/2
			top = (height - new_height)/2
			right = (width + new_width)/2
			bottom = (height + new_height)/2

			# Crop the center of the image
			return(img.crop((left, top, right, bottom)))

		def overlay_mask_part(img_a,img_b,mode):
			if (mode ***REMOVED*** 0):
				img_a = ImageChops.darker(img_a, img_b)
			else: img_a = ImageChops.lighter(img_a, img_b)
			return(img_a)

		def process_mask_parts(these_preds,these_prompt_parts,mode,final_img = None):
			for i in range(these_prompt_parts):
				filename = f"mask_{mode}_{i}.png"
				plt.imsave(filename,torch.sigmoid(these_preds[i][0]))

				# TODO: Figure out how to convert the plot above to numpy instead of re-loading image
				img = cv2.imread(filename)
				gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				(thresh, bw_image) = cv2.threshold(gray_image, mask_precision, 255, cv2.THRESH_BINARY)

				if (mode ***REMOVED*** 0): bw_image = numpy.invert(bw_image)

				if (debug):
					print(f"bw_image: {bw_image}")
					print(f"final_img: {final_img}")

				# overlay mask parts
				bw_image = gray_to_pil(bw_image)
				if (i > 0 or final_img is not None):
					bw_image = overlay_mask_part(bw_image,final_img,mode)

				# For debugging only:
				if (debug): bw_image.save(f"processed_{filename}")

				final_img = bw_image

			return(final_img)

		def load_model_from_config(config, ckpt, verbose=False):
			model = instantiate_from_config(config.model)

			# Load first and second stages of SD v1.4 and CLIP text model
			ldm_weights = torch.load(config.model.params.sd_features_stage_config.params.path, map_location="cpu")["state_dict"]
			
			ldm_weights_keys_updated = {}
			for k, v in ldm_weights.items():
				if '.diffusion_model' in k:
					k = 'sd_features_stage_model.model.'+k
				ldm_weights_keys_updated[k] = v
			m, u = model.load_state_dict(ldm_weights_keys_updated, strict=False) # loading the both stages of LDM along with the clip text encoder 

			print(f"Loading model from {ckpt}")
			pl_sd = torch.load(ckpt, map_location="cpu")
			if "global_step" in pl_sd:
				print(f"Global Step: {pl_sd['global_step']}")
			sd = pl_sd["state_dict"]
			m, u = model.load_state_dict(sd, strict=False) # loading just the segmentation model
			if len(m) > 0 and verbose:
				print("missing keys:")
				print(m)
			if len(u) > 0 and verbose:
				print("unexpected keys:")
				print(u)
			# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
			# num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
			# print("Num of trainable parameters: "+str(num_trainable_params))

			model.cuda()
			model.eval()
			return model

		def get_mask():
			# load model
			config_file = 'extensions/LD-ZNet/configs/ldznet/phrasecut.yaml'
			config = OmegaConf.load(f"{config_file}")
			model = load_model_from_config(config=config, ckpt='extensions/LD-ZNet/checkpoints/LDZNet_txt2mask.ckpt')

			# model.eval();
			# model_dir = "./repositories/clipseg/weights"
			# os.makedirs(model_dir, exist_ok=True)
			# d64_file = f"{model_dir}/rd64-uni.pth"
			# d16_file = f"{model_dir}/rd16-uni.pth"
			# delimiter_string = "|"
			
			# # Download model weights if we don't have them yet
			# if not os.path.exists(d64_file):
			# 	print("Downloading clipseg model weights...")
			# 	download_file(d64_file,"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni.pth")
			# 	download_file(d16_file,"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth")
			# 	# Mirror: 
			# 	# https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth
			# 	# https://github.com/timojl/clipseg/raw/master/weights/rd16-uni.pth
			
			# # non-strict, because we only stored decoder weights (not CLIP weights)
			# model.load_state_dict(torch.load(d64_file, map_location=torch.device('cuda')), strict=False);			

			# transform = transforms.Compose([
			# 	transforms.ToTensor(),
			# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			# 	transforms.Resize((512, 512)),
			# ])
			# img = transform(p.init_images[0]).unsqueeze(0)
			delimiter_string = "|"

			image = p.init_images[0]
			w, h = image.size
			# print(f"loaded input image of size ({w}, {h}) from {path}")
			aspect_ratio = float(h)/w
			new_w = int(math.sqrt(384*384 / aspect_ratio))
			new_h = int(aspect_ratio*new_w)
			new_w, new_h = map(lambda x: x - x % 64, (new_w, new_h))  # resize to integer multiple of 64
			# w, h = 384, 384
			image = image.resize((new_w, new_h), resample=PIL.Image.LANCZOS)
			image = np.array(image).astype(np.float32) / 255.0
			image = image[None].transpose(0, 3, 1, 2)
			image = torch.from_numpy(image).to(torch.device("cuda"))
			init_image = 2.*image - 1.

			prompts = mask_prompt.split(delimiter_string)
			prompt_parts = len(prompts)
			negative_prompts = negative_mask_prompt.split(delimiter_string)
			negative_prompt_parts = len(negative_prompts)

			init_image = repeat(init_image, '1 ... -> b ...', b=1)
			init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
			init_image_arr = (255*(1+init_image[0])/2).cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1]
			t = torch.randint(0, 1, (init_image.shape[0],)).cuda().long() # t should always be 0 for this experiment

			# predict
			with torch.no_grad():
				with model.ema_scope():
					prompts = model.sample_prompts(prompts)
					c = model.get_learned_conditioning(prompts)

					# img_c =model.get_learned_img_conditioning(init_image)
					sd_features = model.encode_sd_features_stage(init_latent, c)#, timestep=400)
					preds = model.apply_model(init_latent, t, c, sd_features)


					negative_prompts = model.sample_prompts(negative_prompts)
					negative_c = model.get_learned_conditioning(negative_prompts)
					negative_preds = model.apply_model(init_latent, t, negative_c, sd_features)
					# preds = model(img.repeat(prompt_parts,1,1,1), prompts)[0]
					# negative_preds = model(img.repeat(negative_prompt_parts,1,1,1), negative_prompts)[0]

			#tests
			if (debug):
				print("Check initial mask vars before processing...")
				print(f"p.image_mask: {p.image_mask}")
				print(f"p.latent_mask: {p.latent_mask}")
				print(f"p.mask_for_overlay: {p.mask_for_overlay}")

			if (brush_mask_mode ***REMOVED*** 1 and p.image_mask is not None):
				final_img = p.image_mask.convert("RGBA")
			else: final_img = None

			# process masking
			final_img = process_mask_parts(preds,prompt_parts,1,final_img)

			# process negative masking
			if (brush_mask_mode ***REMOVED*** 2 and p.image_mask is not None):
				p.image_mask = ImageOps.invert(p.image_mask)
				p.image_mask = p.image_mask.convert("RGBA")
				final_img = overlay_mask_part(final_img,p.image_mask,0)
			if (negative_mask_prompt): final_img = process_mask_parts(negative_preds,negative_prompt_parts,0,final_img)

			# Increase mask size with padding
			if (mask_padding > 0):
				aspect_ratio = p.init_images[0].width / p.init_images[0].height
				new_width = p.init_images[0].width+mask_padding*2
				new_height = round(new_width / aspect_ratio)
				final_img = final_img.resize((new_width,new_height))
				final_img = center_crop(final_img,p.init_images[0].width,p.init_images[0].height)
		
			return (final_img)
						

		# Set up processor parameters correctly
		p.mode = 1
		p.mask_mode = 1
		p.image_mask =  get_mask().resize((p.init_images[0].width,p.init_images[0].height))
		p.mask_for_overlay = p.image_mask
		p.latent_mask = None # fixes inpainting full resolution


		processed = processing.process_images(p)

		if (mask_output):
			processed.images.append(p.image_mask)

		return processed