"""
predefined tool APIs for Omni3D dataset
"""
import time
import base64
import re
from io import BytesIO
from pathlib import Path
import torch
import numpy as np
from PIL import Image

from utils import LLMClient

class PredefinedModules:
    """predefined API tools with actual implementations."""
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_client = LLMClient(config)
        self._modules = {}
        self._trace_buffer = []
        self._initialize_modules()
    
    def _initialize_modules(self):
        self._init_grounding_models()
        self._init_depth_model()
        self._create_module_functions()
    
    def _init_grounding_models(self):
        try:
            from groundingdino.util.inference import load_model
            grounding_configs = {
                "groundingdino_swint_ogc": "GroundingDINO_SwinT_OGC.py",
                "groundingdino_swinb_cogcoor": "GroundingDINO_SwinB_cfg.py"
            }
            grounding_weights = {
                "groundingdino_swint_ogc": "groundingdino_swint_ogc.pth",
                "groundingdino_swinb_cogcoor": "groundingdino_swinb_cogcoor.pth"
            }
            config_file = grounding_configs.get(self.config.grounding_model, "GroundingDINO_SwinT_OGC.py")
            weight_file = grounding_weights.get(self.config.grounding_model, "groundingdino_swint_ogc.pth")
            self.grounding_dino = load_model(
                f"{self.config.models_path}/GroundingDINO/groundingdino/config/{config_file}",
                f"{self.config.models_path}/GroundingDINO/weights/{weight_file}"
            )
            print(f"**SUCCESS**: loaded GroundingDINO: {weight_file}, box_threshold: {self.config.grounding_model_box_threshold}, text_threshold: {self.config.grounding_model_text_threshold}")
        except Exception as e:
            print(f"**BUG**: Could not load GroundingDINO: {e}")
            raise
    
    def _init_depth_model(self):
        try:
            from unidepth.models import UniDepthV2
            depth_models = {
                "unidepth-v2-vits14": "lpiccinelli/unidepth-v2-vits14",
                "unidepth-v2-vitl14": "lpiccinelli/unidepth-v2-vitl14"
            }
            model_name = depth_models.get(self.config.depth_model, "lpiccinelli/unidepth-v2-vits14")
            self.unidepth_model = UniDepthV2.from_pretrained(model_name).to(self.device)
            print(f"**SUCCESS**: loaded UniDepth: {model_name}")
        except Exception as e:
            print(f"**BUG**: Could not load UniDepth model: {e}")
            raise
    
    def _create_module_functions(self):
        self._modules = {
            "loc": self._loc_bbox,
            "vqa": self._vqa_bbox,
            "depth": self._depth_bbox,
            "same_object": self._same_object_bbox,
            "get_2D_object_size": self._get_2d_size_bbox
        }
    
    def get_all(self):
        return self._modules
    
    def get_trace(self):
        return "\n".join(self._trace_buffer)
    
    def clear_trace(self):
        self._trace_buffer = []
    
    def _write_trace(self, html):
        self._trace_buffer.append(html)
    
    def _html_embed_image(self, image, width=None):
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            image = Image.fromarray(image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        width_attr = f'width="{width}"' if width else ''
        return f'<img src="data:image/png;base64,{img_str}" {width_attr}>'
    
    def _box_image(self, image, bboxes):
        """draw bounding boxes on image"""
        import cv2
        img_array = np.array(image)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 8)    # red bbox, linewidth=8
        return Image.fromarray(img_array)
    
    def _loc_bbox(self, image, object_prompt):
        """locate objects returning bounding boxes"""
        self._write_trace(f"<p><b>loc</b>: Locate '{object_prompt}'</p>")
        if isinstance(image, (str, Path)):
            image = Image.open(Path(image)).convert('RGB')
        bboxes = self._grounding_dino_locate(image, object_prompt)
        if len(bboxes) == 0:
            self._write_trace(f"<p>No objects found</p>")
            return []
        boxed_image = self._box_image(image, bboxes)
        self._write_trace(self._html_embed_image(boxed_image, 300))
        self._write_trace(f"<p>{len(bboxes)} {object_prompt} found</p>")
        self._write_trace(f"<p>Boxes: {bboxes}</p>")
        return bboxes
    
    def _grounding_dino_locate(self, image, object_prompt):
        """use groundingdino to locate objects"""
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image.convert("RGB"), None)
        prompt = f"{object_prompt.replace(' ', '-')} ."
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=image_transformed,
                caption=prompt,
                box_threshold=self.config.grounding_model_box_threshold,
                text_threshold=self.config.grounding_model_text_threshold,
                device=self.device,
            )
        width, height = image.size
        bboxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = int((cx - 0.5 * w) * width)
            y1 = int((cy - 0.5 * h) * height)
            x2 = int((cx + 0.5 * w) * width)
            y2 = int((cy + 0.5 * h) * height)
            bboxes.append([x1, y1, x2, y2])
        return bboxes
    
    def _vqa_bbox(self, image, question, bbox):
        """answer questions about object in bbox"""
        self._write_trace(f"<p><b>vqa</b>: Question: {question}</p>")
        if isinstance(image, (str, Path)):
            image = Image.open(Path(image)).convert('RGB')
        if bbox:
            boxed_image = self._box_image(image, bboxes=[bbox])
            answer = self._vqa_with_gpt(boxed_image, question)
            self._write_trace(self._html_embed_image(boxed_image, 300))
        else:   # no valid bbox --> use the whole image
            answer = self._vqa_with_gpt(image, question)
        self._write_trace(f"<p>Answer: {answer}</p>")
        return answer.lower()
    
    def _depth_bbox(self, image, bbox):
        """get depth of bbox region"""
        x_mid = (bbox[0] + bbox[2]) / 2
        y_mid = (bbox[1] + bbox[3]) / 2
        self._write_trace(f"<p><b>depth</b>: Get depth at bbox center: ({x_mid}, {y_mid})</p>")
        if isinstance(image, (str, Path)):
            image = Image.open(Path(image)).convert('RGB')
        with torch.no_grad():
            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(self.device)
            preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()
            depth_value = float(preds[int(y_mid), int(x_mid)])
        boxed_image = self._box_image(image, [bbox])
        self._write_trace(self._html_embed_image(boxed_image, 300))
        self._write_trace(f"<p>Depth: {depth_value}</p>")
        return depth_value
    
    def _same_object_bbox(self, image, bbox1, bbox2):
        """check if two bboxes are the same object"""
        self._write_trace(f"<p><b>same_object</b>: bbox1={bbox1}, bbox2={bbox2}</p>")
        if isinstance(image, (str, Path)):
            image = Image.open(Path(image)).convert('RGB')
        iou = self._calculate_iou(bbox1, bbox2)
        result = iou > 0.92
        boxed_image = self._box_image(image, [bbox1, bbox2])
        self._write_trace(self._html_embed_image(boxed_image, 300))
        self._write_trace(f"<p>IoU: {iou:.3f}, Same: {result}</p>")
        return result
    
    def _get_2d_size_bbox(self, image, bbox):
        """get 2D size of bbox"""
        width = abs(bbox[2] - bbox[0])
        height = abs(bbox[3] - bbox[1])
        self._write_trace(f"<p><b>get_2D_object_size</b>: Get 2D Object Size for bbox: {bbox}</p>")
        if isinstance(image, (str, Path)):
            image = Image.open(Path(image)).convert('RGB')
        boxed_image = self._box_image(image, [bbox])
        self._write_trace(self._html_embed_image(boxed_image, 300))
        self._write_trace(f"<p>Width: {width}, Height: {height}</p>")
        return (width, height)
    
    def _vqa_with_gpt(self, image, question):
        vqa_prompt = f"You will be shown an image with a red bounding box and asked to answer a question based on the object inside the bounding box. Please only answer with regards to the object IN the bounding box. Answer this question with one word based on the object in the bounding box and put your answer in between <answer></answer> tags: {question}"
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        vqa_max_retries = 3
        for attempt in range(vqa_max_retries):
            try:
                answer_text = self.llm_client.generate_with_image(
                    vqa_prompt, 
                    img_base64, 
                    self.config.vqa_model, 
                    temperature=0.0)
                match = re.search(r'<answer>(.*?)</answer>', answer_text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip().lower()
                else:
                    return answer_text.strip().lower()  # fallback if no tags found
            except Exception as e:
                print(f"VQA - API call failed (vqa attempt {attempt + 1}/{vqa_max_retries}): {e}")
                if attempt < vqa_max_retries - 1:
                    wait_time = 0.1 * (attempt+1)
                    time.sleep(wait_time)
                else:
                    print(f"VQA - Error after {vqa_max_retries} vqa attempts: {e}")
                    return "unknown"
        return "unknown"  # fallback if all retries fail

    def _calculate_iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)
        area_inter = width_inter * height_inter
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        area_union = area_box1 + area_box2 - area_inter
        return area_inter / area_union if area_union != 0 else 0
