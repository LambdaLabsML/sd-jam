from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import gradio as gr
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import utils
import datetime
import time
import psutil
import random
import string
import re

from ray.serve.gradio_integrations import GradioServer

start_time = time.time()
is_colab = utils.is_google_colab()
result_path = "/home/ubuntu/ml/chuan/lambda-demo/results/"
max_model_name_length = 16

class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None

models = [
     Model("Anything V3", "Linaqruf/anything-v3.0", ""),
     Model("Midjourney v4", "prompthero/midjourney-v4-diffusion", "mdjrny-v4 style "),
     Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
     Model("Redshift", "nitrosocke/redshift-diffusion", "redshift style "),
     Model("Analog Diffusion", "wavymulder/Analog-Diffusion", "analog style "),
     Model("Wavyfusion", "wavymulder/wavyfusion", "wa-vy style "),
     Model("Pokémon", "lambdalabs/sd-pokemon-diffusers"),
     Model("Naruto", "lambdalabs/sd-naruto-diffusers"),
     Model("Modern Disney", "nitrosocke/mo-di-diffusion", "modern disney style "),
     Model("Classic Disney", "nitrosocke/classic-anim-diffusion", "classic disney style "),
     Model("Van Gogh", "dallinmackay/Van-Gogh-diffusion", "lvngvncnt "),
     Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
     Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
     Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy "),
     Model("Robo Diffusion", "nousr/robo-diffusion"),
     Model("Arcane", "nitrosocke/Arcane-Diffusion", "arcane style "),
     Model("Archer", "nitrosocke/archer-diffusion", "archer style "),
     Model("Cyberpunk Anime", "DGSpitzer/Cyberpunk-Anime-Diffusion", "dgs illustration style "),
  ]

custom_model = None
if is_colab:
  models.insert(0, Model("Custom model"))
  custom_model = models[0]

last_mode = "txt2img"
current_model = models[1] if is_colab else models[0]
current_model_path = current_model.path

if is_colab:
  pipe = StableDiffusionPipeline.from_pretrained(
      current_model.path,
      torch_dtype=torch.float16,
      scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
      safety_checker=lambda images, clip_input: (images, False)
      )

else:
  pipe = StableDiffusionPipeline.from_pretrained(
      current_model.path,
      torch_dtype=torch.float16,
      scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
      )
    
if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  pipe.enable_xformers_memory_efficient_attention()

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def on_model_change(model_name):
  
  prefix = "Enter prompt. \"" + next((m.prefix for m in models if m.name == model_name), None) + "\" is prefixed automatically" if model_name != models[0].name else "Don't forget to use the custom model prefix in the prompt!"

  return gr.update(visible = model_name == models[0].name), gr.update(placeholder=prefix)

def inference(model_name, prompt, guidance, steps, n_images=1, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):

  print(psutil.virtual_memory()) # print memory usage

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None

  

  try:
    if img is not None:
      images = img_to_img(model_path, prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator)
    else:
      images = txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator)

    # save images to disk
    run_id = ''.join(random.choices(string.ascii_lowercase, k=5))
    download_file_path = [ ]
    for i_img in range(n_images):
      model_name_clean = model_name.replace(' ', '')
      model_name_clean = re.sub(r'[\\/*?:"<>|]',"", model_name_clean)
      model_name_clean = model_name_clean[:max_model_name_length]

      image_name = result_path + \
          model_name_clean + \
          "_" + run_id + \
          "_" + str(i_img) + ".png"

      metadata = PngInfo()
      metadata.add_text("stablediffusion.Style", model_name)
      metadata.add_text("stablediffusion.Prompt", prompt)
      metadata.add_text("stablediffusion.Steps", str(steps))
      metadata.add_text("stablediffusion.Guidance", str(guidance))
      metadata.add_text("stablediffusion.Seed", str(seed))

      images[i_img].save(image_name, pnginfo=metadata)
      download_file_path.append(image_name)

    return images, download_file_path, None
  except Exception as e:
    return None, error_str(e)


def inference_examples(model_name, prompt, guidance, steps, n_images=1, width=512, height=512, seed=0, neg_prompt=""):

  print(psutil.virtual_memory()) # print memory usage

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None


  try:
    images = txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator)

    # save images to disk
    run_id = ''.join(random.choices(string.ascii_lowercase, k=5))
    for i_img in range(n_images):
      model_name_clean = model_name.replace(' ', '')
      model_name_clean = re.sub(r'[\\/*?:"<>|]',"", model_name_clean)
      model_name_clean = model_name_clean[:max_model_name_length]

      image_name = result_path + \
          model_name_clean + \
          "_" + run_id + \
          "_" + str(i_img) + ".png"

      metadata = PngInfo()
      metadata.add_text("stablediffusion.Style", model_name)
      metadata.add_text("stablediffusion.Prompt", prompt)
      metadata.add_text("stablediffusion.Steps", str(steps))
      metadata.add_text("stablediffusion.Guidance", str(guidance))
      metadata.add_text("stablediffusion.Seed", str(seed))

      images[i_img].save(image_name, pnginfo=metadata)
    return images
  except Exception as e:
    return None, error_str(e)

def txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator):

    print(f"{datetime.datetime.now()} txt_to_img, model: {current_model.name}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
          pipe = StableDiffusionPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
              safety_checker=lambda images, clip_input: (images, False)
              )
        else:
          pipe = StableDiffusionPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
              )
          # pipe = pipe.to("cpu")
          # pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
          pipe.enable_xformers_memory_efficient_attention()
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt  
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    #return replace_nsfw_images(result)
    return result.images

def img_to_img(model_path, prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator):

    print(f"{datetime.datetime.now()} img_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if is_colab or current_model == custom_model:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
              safety_checker=lambda images, clip_input: (images, False)
              )
        else:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.float16,
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
              )
          # pipe = pipe.to("cpu")
          # pipe = current_model.pipe_i2i
        
        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
          pipe.enable_xformers_memory_efficient_attention()
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt = neg_prompt,
        num_images_per_prompt=n_images,
        image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        # width = width,
        # height = height,
        generator = generator)
        
    # return replace_nsfw_images(result)
    return result.images

def replace_nsfw_images(results):

    if is_colab:
      return results.images
      
    for i in range(len(results.images)):
      if results.nsfw_content_detected[i]:
        results.images[i] = Image.open("nsfw.png")
    return results.images

css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:20rem}
"""

demo = gr.Blocks(css=css)

with demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Finetuned Diffusion</h1>
              </div>
              <p>
               Demo for multiple fine-tuned Stable Diffusion models, trained on different styles: <br>
               <a href="https://huggingface.co/Linaqruf/anything-v3.0">Anything V3</a>,
               <a href="https://huggingface.co/prompthero/midjourney-v4-diffusion">Midjourney v4 style</a>,
               <a href="https://huggingface.co/lambdalabs/sd-pokemon-diffusers">Pokémon</a>,
               <a href="https://huggingface.co/lambdalabs/sd-naruto-diffusers">Naruto</a>,
               <a href="https://huggingface.co/DGSpitzer/Cyberpunk-Anime-Diffusion">Cyberpunk Anime</a>, 
               <a href="https://huggingface.co/nitrosocke/elden-ring-diffusion">Elden Ring</a>, 
               <a href="https://huggingface.co/nitrosocke/redshift-diffusion">Redshift renderer (Cinema4D)</a>, 
               <a href="https://huggingface.co/nitrosocke/mo-di-diffusion">Modern Disney</a>, 
               <a href="https://huggingface.co/nitrosocke/classic-anim-diffusion">Classic Disney</a>, 
               <a href="https://huggingface.co/wavymulder/Analog-Diffusion">Analog Diffusion</a>,
               <a href="https://huggingface.co/dallinmackay/Van-Gogh-diffusion">Loving Vincent (Van Gogh)</a>, 
               <a href="https://huggingface.co/wavymulder/wavyfusion">Wavyfusion</a>,
               <a href="https://huggingface.co/naclbit/trinart_stable_diffusion_v2">TrinArt v2</a>,
               <a href="https://huggingface.co/nitrosocke/spider-verse-diffusion">Spider-Verse</a>, 
               <a href="https://huggingface.co/Fictiverse/Stable_Diffusion_BalloonArt_Model">Balloon Art</a>,
               <a href="https://huggingface.co/dallinmackay/Tron-Legacy-diffusion">Tron Legacy</a>,
               <a href="https://huggingface.co/AstraliteHeart/pony-diffusion">Pony Diffusion</a>, 
               <a href="https://huggingface.co/nousr/robo-diffusion">Robo Diffusion</a>,
               <a href="https://huggingface.co/nitrosocke/Arcane-Diffusion">Arcane</a>, 
               <a href="https://huggingface.co/nitrosocke/archer-diffusion">Archer</a>.
              </p>
            </div>
        """
    )
    with gr.Row():
        
        with gr.Column(scale=55):
          with gr.Group():
              model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
              with gr.Box(visible=False) as custom_model_group:
                custom_model_path = gr.Textbox(label="Custom model path", placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion", interactive=True)
                gr.HTML("<div><font size='2'>Custom models have to be downloaded first, so give it some time.</font></div>")
              
              with gr.Row():
                prompt = gr.Textbox(label="Prompt", show_label=False, max_lines=2,placeholder="Enter prompt. Style applied automatically").style(container=False)
                generate = gr.Button(value="Generate").style(rounded=(False, True, True, False))


              # image_out = gr.Image(height=512)
              gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

          error_output = gr.Markdown()
          downloads=gr.Files(label="downloads")

        with gr.Column(scale=45):
          with gr.Tab("Options"):
            with gr.Group():
              neg_prompt = gr.Textbox(label="Negative prompt", placeholder="What to exclude from the image")

              n_images = gr.Slider(label="Images", value=1, minimum=1, maximum=4, step=1)

              with gr.Row():
                guidance = gr.Slider(label="Guidance scale", value=7.5, maximum=15)
                steps = gr.Slider(label="Steps", value=50, minimum=2, maximum=75, step=1)

              with gr.Row():
                width = gr.Slider(label="Width", value=512, minimum=64, maximum=1024, step=8)
                height = gr.Slider(label="Height", value=512, minimum=64, maximum=1024, step=8)

              seed = gr.Slider(0, 2147483647, label='Seed (0 = random)', value=0, step=1)

          with gr.Tab("Image to image"):
              with gr.Group():
                image = gr.Image(label="Image", height=256, tool="editor", type="pil")
                strength = gr.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5)

    if is_colab:
      model_name.change(on_model_change, inputs=model_name, outputs=[custom_model_group, prompt], queue=False)
      custom_model_path.change(custom_model_changed, inputs=custom_model_path, outputs=None)
    # n_images.change(lambda n: gr.Gallery().style(grid=[2 if n > 1 else 1], height="auto"), inputs=n_images, outputs=gallery)

    inputs = [model_name, prompt, guidance, steps, n_images, width, height, seed, image, strength, neg_prompt]
    outputs = [gallery, downloads, error_output]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

    ex = gr.Examples([
        [models[0].name, "scenery, shibuya tokyo, post-apocalypse, ruins, rust, sky, skyscraper, abandoned, blue sky, broken window, building, cloud, crane machine, outdoors, overgrown, pillar, sunset", 7.5, 50, 1, 1024, 1024, 94107, ""],
        [models[1].name, "massive rocket ship during its launch from the ground| ground level full of fire and smoke| extreme long shot| centered| vibrant| massive scale| dynamic lighting| in mdjrny-v4 style", 7.5, 50, 1, 1024, 1024, 94107, ""],
        [models[2].name, "elden ring style dark blue night (castle) on a cliff", 7.0, 30, 1, 1024, 576, 94107, "bright day"],
        [models[3].name, "redshift style magical princess with golden hair", 7.0, 50, 1, 768, 768, 94107, ""],
        [models[4].name, "analog style portrait of Heath Ledger as a 1930s baseball player", 7.0, 20, 1, 512, 768, 94107, "blur haze"],
        [models[5].name, "wa-vy style cute girl at the (lantern festival:1.2)", 7.0, 50, 1, 768, 768, 94107, "blurry face"],
        [models[6].name, "Cute Baby Yoda creature", 7.5, 50, 1, 512, 512, 94107, "Vivillon"],
        [models[7].name, "ninja bunny portrait", 7.5, 50, 1, 512, 512, 94107, "mask"],
        [models[8].name, "Anime fine details portrait of a magical princess in front of modern tokyo city landscape, anime masterpiece, 8k, sharp high quality", 7.5, 50, 1, 768, 768, 94107, ""],
        [models[9].name, "Anime fine details portrait of a magical princess in front of modern tokyo city landscape, anime masterpiece, 8k, sharp high quality", 7.5, 50, 1, 768, 768, 94107, ""],
        [models[10].name, "lvngvncnt, streets and canals in old town Amsterdam, highly detailed, highly detailed", 7.5, 50, 1, 768, 768, 94107, ""],
    ], inputs=[model_name, prompt, guidance, steps, n_images, width, height, seed, neg_prompt], outputs=gallery, fn=inference_examples, cache_examples=True)

    gr.HTML("""
    <div style="border-top: 1px solid #303030;">
      <br>
      <p>Space originally created by:<br>
      <a href="https://twitter.com/hahahahohohe"><img src="https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social" alt="Twitter Follow"></a><br>
      <p>Customized by:<br>
      <a href="https://twitter.com/LambdaAPI">mlteam@LambdaAPI</a>
    </div>
    """)

print(f"Space built in {time.time() - start_time:.2f} seconds")


#demo.queue(concurrency_count=1)
#demo.launch(debug=True, share=True)

app = GradioServer.options(num_replicas=torch.cuda.device_count(), ray_actor_options={"num_gpus" : 1.0}).bind(demo)
