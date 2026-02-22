import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
import warnings
from huggingface_hub import snapshot_download

##############################################
##############################################
##############################################
# added for vllm
from ..dataset import DATASET_MODALITY
import math

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')

def _resize_image(image, max_side):
    resize_scale = max_side / max(image.size)
    new_size = (
        int(image.size[0] * resize_scale),
        int(image.size[1] * resize_scale),
    )
    return image.resize(new_size)

def _rgba_to_rgb(image):
    from PIL import Image
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")

def _encode_image(image, image_format):
    from io import BytesIO
    with BytesIO() as output:
        image.convert("RGB").save(output, format=image_format)
        import base64
        base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
    return base64_encoded_data

def encode_image(image_path, max_side=None):
    from mimetypes import guess_type
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

    from PIL import Image
    image = Image.open(image_path)
    # Handle the alpha channel
    if image.mode == "RGBA":
        image = _rgba_to_rgb(image)
    if max_side:
        image = _resize_image(image, max_side)
    encoded_image = _encode_image(image, image_format)

    return encoded_image, mime_type

def create_image_content(image_path, min_pixels, max_pixels):
    base64_image, mime_type = encode_image(image_path)
    return {
        "type": "image",
        "image": f"data:{mime_type};base64,{base64_image}",
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    }

def process_video(video_path, num_frames, min_pixels, max_pixels):
    import cv2
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second

    # the sampling rate using max number of frames
    sampling_gap_maxframe = (
        1 if not num_frames else math.ceil(frame_count / num_frames)
    )
    sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)

    frame_number = 0
    images = []

    while True:
        import tempfile
        success, frame = cap.read()
        if not success:
            break
        # Sample frames based on the dynamic sampling rate
        if frame_number % sampling_gap == 0:
            # Create a temporary file for the frame
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False
            ) as temp_frame:
                cv2.imwrite(temp_frame.name, frame)
                images.append(create_image_content(temp_frame.name, min_pixels, max_pixels))
                os.remove(temp_frame.name)
        frame_number += 1
    if frame_number == 0:
        raise ValueError(f"Failed to read video from {video_path}, check data...")
    logging.info(
        f"Sampled {len(images)}/{frame_number} frames from video {video_path}"
    )
    cap.release()
    return images
##############################################
##############################################
##############################################

class Pixtral(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='mistralai/Pixtral-12B-2409', **kwargs):

        self.model_path = model_path
        self.kwargs = kwargs

        if os.path.exists(model_path):
            cache_path = model_path
        else:
            if get_cache_path(model_path) is None:
                snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(self.model_path, repo_type='models')

        if kwargs.get("use_vllm", False):
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            self.system_prompt = None # only QVQ-72B-Preview needs one
            tp_size = 1 if gpu_count == 0 else min(8, gpu_count)
            logging.info(f"Using vLLM for {self.model_path} with {tp_size} GPUs")

            self.llm = LLM(
                model=self.model_path,
                # max_model_len=32768,
                
                # max_model_len=45056,
                # max_num_batched_tokens=45056,
                # num_gpu_blocks_override=2816,
                max_model_len=38432,
                max_num_batched_tokens=38432,
                num_gpu_blocks_override=2402,

                tokenizer_mode="mistral",
                # allowed_local_media_path="/srv/muse-lab/datasets/VLMEvalKitdata/LMUData/images",
                allowed_local_media_path="/srv/muse-lab/datasets/VLMEvalKitdata",
                
                # from Konstantinos
                # limit_mm_per_prompt = 64,
                # swap_space=args.swap_space,
                # limit_mm_per_prompt = {“image”: 64}
                # scheduling_policy=approach.scheduling_policy,
                # disable_log_stats=False,
                # hf_token=True, # requires huggingface-cli login
                # hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]} if model.alias.startswith("deepseek-vl2") else None,
                # disable_mm_preprocessor_cache=True,
                # max_num_seqs=1

                tensor_parallel_size=tp_size,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            )

            self.use_vllm = True
        else:
            # fall back to mistral-inference
            try:
                from mistral_inference.transformer import Transformer
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            except ImportError as err:
                logging.critical('Please install `mistral-inference` and `mistral_common`')
                raise err
            self.use_vllm = False

            self.tokenizer = MistralTokenizer.from_file(f'{cache_path}/tekken.json')
            model = Transformer.from_folder(cache_path, device='cpu')
            model.cuda()
            self.model = model
        self.max_tokens = 2048
        torch.cuda.empty_cache()
        # ##############################################
        # model = Transformer.from_folder(
        #     cache_path,
        #     device='cuda',
        #     dtype=torch.bfloat16,   # cut VRAM by ~50% # Recommended on A100/H100 GPUs.
        #     # dtype=torch.float16,   # cut VRAM by ~50%
        #     softmax_fp32=True       # keep logits stable in fp32
        #     # softmax_fp32=False
        #     )

    def generate_inner_transformers(self, message, dataset=None):
    # def generate_inner(self, message, dataset=None):
        try:
            from mistral_inference.generate import generate
            from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        msg_new = []
        for msg in message:
            tp, val = msg['type'], msg['value']
            if tp == 'text':
                ############################################################################
                # changes for MCQ - like in vlmeval/vlm/qwen2_vl/prompt.py
                ############################################################################
                # SOS
                if any(s in dataset for s in ["MMBench_DEV_EN", "Video-MME", "Video_MME"]):
                    val = val + " Please select the correct answer from the options above. Respond with only the letter (A, B, C, or D) of the correct option."
                # SOS
                ############################################################################
                msg_new.append(TextChunk(text=val))
            elif tp == 'image':
                b64 = encode_image_file_to_base64(val)
                image_url = f'data:image/jpeg;base64,{b64}'
                msg_new.append(ImageURLChunk(image_url=image_url))

        completion_request = ChatCompletionRequest(messages=[UserMessage(content=msg_new)])
        encoded = self.tokenizer.encode_chat_completion(completion_request)
        images = encoded.images
        tokens = encoded.tokens

        out_tokens, _ = generate(
            [tokens],
            self.model,
            images=[images],
            max_tokens=self.max_tokens,
            temperature=0,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)

        result = self.tokenizer.decode(out_tokens[0])
        return result

#################################################################################################################
#################################################################################################################
#################################################################################################################
    # added for vllm

    def _prepare_content_vllm(self, inputs, dataset=None):
        content = []
        for s in inputs:
            if s['type'] == 'text':
                content.append({"type": "text", "text": s['value']})
            # elif s['type'] == 'video':
            #     video_frames = process_video(s['value'], self.nframe or 32)  # you’ll need process_video util
            #     content.append({"type": "video", "video": video_frames})
            # elif s['type'] == 'video':
            #     # Extract n frames
            #     video_frames = process_video(s['value'], self.nframe or 16)  # default 16
            #     # Convert frames into image_url messages
            #     for frame in video_frames:
            #         content.append({
            #             "type": "image_url",
            #             "image_url": {"url": ensure_image_url(frame)}
            #         })

            elif s['type'] == 'image':
                # content.append({"type": "image", "image": ensure_image_url(s['value'])})
                content.append({"type": "image_url",
                                "image_url": {
                                    "url": ensure_image_url(s['value'])
                                    }})
            else:
                raise ValueError(f"Unsupported type {s['type']}")
        return content

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._prepare_content_vllm(message, dataset)})

        # # Convert to chat text with processor (or tokenizer prompt)
        # ###############################
        # from transformers import AutoProcessor
        # self.processor = AutoProcessor.from_pretrained(self.model_path)
        # ###############################
        # prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens)

        # inputs = {"prompt": prompt}
        # if dataset and "VIDEO" in DATASET_MODALITY(dataset):
        #     # pass frames to vLLM
        #     frame_paths = []
        #     for i in messages[-1]['content']:
        #         if i['type'] == 'image':
        #             frame_paths.append(i['image'].split('file://')[-1])
        #     img_list = []
        #     for pth in frame_paths:
        #         img = Image.open(pth).convert("RGB")
        #         img_list.append(img)
        #     inputs["multi_modal_data"] = {"image": img_list}
            
        #     # video_frames = messages[-1]['content'][-1]['video']
        # elif any(s['type'] == 'image' for s in message):
        #     inputs["multi_modal_data"] = {"image": [...]}

        # outputs = self.llm.generate(inputs, sampling_params)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        else:
            return self.generate_inner_transformers(message, dataset=dataset)

#################################################################################################################
#################################################################################################################
#################################################################################################################

    def use_custom_prompt(self, dataset: str) -> bool:
        if DATASET_MODALITY(dataset) == 'VIDEO':
            return True
        return False

    def resize_proportional(self, image: Image.Image, scale: int = 20) -> Image.Image:
        """
        Proportionally resizes an image by reducing both dimensions by `scale` percent.
        Ensures output dimensions are even numbers (rounded up).
        
        Parameters
        ----------
        image : PIL.Image
            Input image.
        scale : int
            Percentage reduction (e.g. 20 → shrink by 20%). 
            Can also be negative for enlargement (e.g. -20 → enlarge by 20%).
        """
        w, h = image.size

        # convert percentage into factor
        factor = 1 - (scale / 100.0)

        new_w = int(w * factor)
        new_h = int(h * factor)

        # round up to even numbers
        if new_w % 2 != 0:
            new_w += 1
        if new_h % 2 != 0:
            new_h += 1

        return image.resize((new_w, new_h), Image.LANCZOS)

    def smart_resize(
        self,
        height: int,
        width: int,
        image: Image.Image,
        factor: int = 28, # IMAGE_FACTOR = 28
        min_pixels: int = 4 * 28 * 28, # MIN_PIXELS = 4 * 28 * 28
        # max_pixels: int = 16384 * 28 * 28, # MAX_PIXELS = 16384 * 28 * 28
        max_pixels: int = 150800, # MAX_PIXELS by 520 * 290
    ) -> tuple[int, int]:
        
        """
        Based on .env_image_mc/lib/python3.11/site-packages/qwen_vl_utils/vision_process.py

        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """

        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor

        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor
        
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        
        return image.resize((w_bar, h_bar), Image.LANCZOS)

    def save_video_frames(self, video_id):        
        """Extract frames uniformly or by fps, save to disk, and return paths + info."""

        # # RESIZED_SIZE = 60 # -> (512, 288) for Video-MME
        # RESIZED_SIZE = 62 # -> (512, 288) for MMBench-Video
        
        import decord
        # MAX_PIXELS = 150800 # MAX_PIXELS by 520 * 290
        # MAX_PIXELS = 149184 # MAX_PIXELS by 518 * 288
        # MAX_PIXELS = 141224 # MAX_PIXELS by 508 * 278
        MAX_PIXELS = 138096 # MAX_PIXELS by 504 * 274
        if self.dataset_name == "TempCompass_Captioning":
            line = video_id
            vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
            video_id = line['video']
            # MAX_PIXELS = 135000 # MAX_PIXELS by 500 * 270
            MAX_PIXELS = 142800 # MAX_PIXELS by 510 * 280
        else:
            if self.data_root.endswith("/video"):
                # for MMBench-Video
                vid_path = osp.join(self.data_root, video_id + ".mp4")
            else:
                # for Video-MME
                vid_path = osp.join(self.data_root, "video", video_id + ".mp4")
        vr = decord.VideoReader(vid_path)
        video_info = {"fps": vr.get_avg_fps(), "n_frames": len(vr)}

        # -------------------------------------------------
        # 1) Use clever sampling if available
        if hasattr(self, "kwargs") and "clever_sampling" in self.kwargs:
            sampling_technique = self.kwargs["clever_sampling"]
            from giannis_stuff.giannis_utils import apply_clever_sampling
            frames_numpy, frame_times, video_time = apply_clever_sampling(
                sampling_technique,
                vid_path,
                max_frames=self.kwargs.get("max_frames", self.nframe),
                sampling_extra_params=self.kwargs.get("sampling_extra_param", {})
            )

            # returns the frames as numpy - I will resize them here
            indices = list(range(len(frames_numpy)))
            frame_paths = [
                # osp.join(self.data_root, "video_frames", video_id, f"frame-{i}-{sampling_technique}-{RESIZED_SIZE}.jpg")
                osp.join(self.data_root, "video_frames", video_id, f"frame-{i}-{sampling_technique}-clever_resized.jpg")
                for i in indices
            ]

            # they have returned not resized and I resize them If I want and then save them
            if not np.all([osp.exists(p) for p in frame_paths]):
                os.makedirs(osp.dirname(frame_paths[0]), exist_ok=True)
                # images = [self.resize_proportional(Image.fromarray(arr), RESIZED_SIZE) for arr in frames_numpy]
                images = [
                    self.smart_resize(
                        image=Image.fromarray(arr), 
                        height=arr.shape[0],
                        width=arr.shape[1],
                        max_pixels=MAX_PIXELS
                    ) for arr in frames_numpy
                    ]

                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths, indices, video_info

        # -------------------------------------------------
        # 2) Otherwise do uniform or fps-based sampling
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vr) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_id)
        elif self.fps > 0:
            total_duration = video_info["n_frames"] / video_info["fps"]
            required_frames = int(total_duration * self.fps)
            step_size = video_info["fps"] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = [
                osp.join(self.data_root, "video_frames", video_id, f"frame-{i}.jpg")
                for i in range(len(indices))
            ]
        else:
            raise ValueError("Either nframe > 0 or fps > 0 must be set.")

        # -------------------------------------------------
        # 3) Extract + save frames if missing
        if not np.all([osp.exists(p) for p in frame_paths]):
            os.makedirs(osp.dirname(frame_paths[0]), exist_ok=True)
            images = [vr[i].asnumpy() for i in indices]
            
            # images = [Image.fromarray(arr) for arr in images]
            # images = [self.resize_proportional(im, scale=RESIZED_SIZE) for im in images]
            images = [
                    self.smart_resize(
                        image=Image.fromarray(im), 
                        height=im.shape[0],
                        width=im.shape[1],
                        max_pixels=MAX_PIXELS
                    ) for im in images
                    ]
            
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, indices, video_info
    
    def build_prompt(self, line, dataset, video_llm=False):

        self.data_root = dataset.data_root
        self.nframe = dataset.nframe
        self.fps = dataset.fps
        self.frame_paths = dataset.frame_paths # function load from VideoMME class and MMBench-Video
        self.dataset_name = dataset.dataset_name

        if dataset.dataset_name == "Video-MME":
            return self.build_prompt_videomme(line, dataset, video_llm)
        elif dataset.dataset_name == "MMBench-Video":
            return self.build_prompt_mmbench_video(line, dataset, video_llm)
        elif dataset.dataset_name == "TempCompass_Captioning":
            return self.build_prompt_tempcompass(line, dataset, video_llm)
        else:
            raise NotImplementedError
    
    def qa_template(self, data):
        question = data['question']
        answer = data['answer']
        return question, answer
    
    def save_video_into_images(self, line):
        frame_paths, _, _ = self.save_video_frames(line)
        return frame_paths

    def build_prompt_tempcompass(self, line, dataset, video_llm=False):
        question, _ = self.qa_template(line)
        message = []
        # video_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        video_path = osp.join(self.data_root, line['prefix'].split("./")[-1], line['video'] + line['suffix'])
        
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value=question))
        return message
    
    def build_prompt_mmbench_video(self, line, dataset, video_llm=False):

        self.FRAMES_TMPL_NOPACK = """
        You will be provided with {} separate frames uniformly sampled from a video, \
        the frames are provided in chronological order of the video.
        Please analyze these images and provide the answer to the question about the video content.
        Please directly reply with your response to the only question.
        """
        frames, indices, video_info = self.save_video_frames(line['video'])

        sys_prompt = self.FRAMES_TMPL_NOPACK.format(len(frames))
        message = [dict(type='text', value=sys_prompt)]
        
        for im in frames:
            message.append(dict(type='image', value=im))
        
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        
        return message
    
    def build_prompt_videomme(self, line, dataset, video_llm=False):
    # def build_prompt(self, line, dataset, video_llm=False):
        """Build vLLM chat messages for Pixtral from a dataset line containing video+question."""
        
        video_id = line["video"]
        question = line["question"]
        candidates = line["candidates"]
        
        self.FRAMES_TMPL_NOSUB = """
        These are the frames of a video. \
        Select the best answer to the following multiple-choice question based on the video. \
        Respond with only the letter (A, B, C, or D) of the correct option.
        """

        frames, indices, video_info = self.save_video_frames(video_id)

        # Messages in Pixtral vLLM format
        self.SYS = ''
        message = [dict(type='text', value=self.SYS)]

        # Add frames as image_urls
        for f in frames:
            message.append(dict(type='image', value=f))

        text_prompt = (self.FRAMES_TMPL_NOSUB)

        message.append(dict(type='text', value=text_prompt))
        question += '\n' + '\n'.join(eval(candidates))
        prompt = 'Question: {}\nAnswer: '.format(question)
        message.append(dict(type='text', value=prompt))
        return message