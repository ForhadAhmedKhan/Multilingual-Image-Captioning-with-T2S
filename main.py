import asyncio, time, uuid, os, cv2, numpy as np, torch, hashlib, re , nltk 
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    Blip2Processor, Blip2ForConditionalGeneration
)
from deepface import DeepFace
from googletrans import Translator
from gtts import gTTS
from num2words import num2words
from typing import List
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from reference_captions import reference_captions
from nltk.translate.bleu_score import corpus_bleu



app = FastAPI()
executor = ThreadPoolExecutor()
os.makedirs("outputs", exist_ok=True)
image_cache = {}



# Load models
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",
    device_map="cpu",  # Ensure CPU usage
    torch_dtype=torch.float32,  # Use float32 for compatibility
    low_cpu_mem_usage=True  # Minimize RAM usage
)
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")



@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("front.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

def convert_numbers_to_words(text):
    def replace_match(match):
        num = match.group()
        try:
            return num2words(int(num))
        except:
            return num  # fallback in case conversion fails
    return re.sub(r'\d+', replace_match, text)

def convert_numbers_to_local_digits(text, lang):
    numeral_maps = {
        "bn": str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"),
        "hi": str.maketrans("0123456789", "०१२३४५६७८९"),
        "zh-cn": str.maketrans("0123456789", "〇一二三四五六七八九"),
        "en": {}
    }
    return text.translate(numeral_maps.get(lang, {}))

def categorize_age_group(age):
    if age < 13:
        return "Child"
    elif 13 <= age < 20:
        return "Teen"
    elif 20 <= age < 50:
        return "Adult"
    else:
        return "Senior"

def resize_image_if_needed(image):
    max_size = 850
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size))
    return image

def determine_position(x, image_width):
    center = image_width / 2
    if x < center - image_width * 0.1:
        return "left"
    elif x > center + image_width * 0.1:
        return "right"
    else:
        return "center"

def build_face_summary(face_info_list, image_width):
    group_counter = {}
    summaries = []
    for face in sorted(face_info_list, key=lambda f: f["position"][0]):
        age_group = face["age_group"]
        emotion = face["emotion"]
        age = face["age"]
        x = face["position"][0]

        group_counter.setdefault(age_group, 0)
        group_counter[age_group] += 1
        label = f"{age_group.lower()}"
        if group_counter[age_group] > 1:
            label += f" {group_counter[age_group]}"

        position = determine_position(x, image_width)
        summaries.append(f"A {age_group.lower()} is on the {position} side, feeling {emotion.lower()} ({age} years old)")

    return ", ".join(summaries)

def generate_caption(image: Image.Image, model_choice: str):
    image = resize_image_if_needed(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        if model_choice == "vit-gpt2":
            pixel_values = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)
            output_ids = vit_model.generate(pixel_values, max_length=16, num_beams=4)
            return vit_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elif model_choice == "blip2":
            inputs = blip2_processor(images=image, return_tensors="pt").to("cpu")
            outputs = blip2_model.generate(**inputs)
            return blip2_processor.decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = blip_processor(image, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs)
            return blip_processor.decode(out[0], skip_special_tokens=True)


def evaluate_caption(filename, generated_caption):

    key = os.path.basename(filename)
    refs = reference_captions.get(key)
    if not refs:
        return None, None

    if any(isinstance(ref, list) for ref in refs):
        refs = [" ".join(ref) for ref in refs]
        
    # Tokenize both reference and hypothesis
    reference_tokens_list = [word_tokenize(ref.lower()) for ref in refs]
    hypothesis_tokens = word_tokenize(generated_caption.lower())

    # Call METEOR with token lists
    meteor_score = round(single_meteor_score(reference_tokens_list[0], hypothesis_tokens), 4)
    
    bleu_score = round(corpus_bleu([reference_tokens_list], [hypothesis_tokens]), 4)

    return meteor_score, bleu_score


def process_image(content, filename, lang, model_choice):
    

    file_hash = hashlib.sha256(content).hexdigest()
    cached_entry = image_cache.get(file_hash, {})
    model_cache = cached_entry.get("languages", {}).get(lang, {}).get(model_choice)
    
    if model_cache:
        return {
            "caption": model_cache["caption"],
            "audio_url": model_cache["audio_url"],
            "image_url": cached_entry["annotated_image_url"],
            "emotions": cached_entry["emotions"],
            "age_groups": cached_entry["age_groups"],
            "model_time_only": model_cache.get("model_time_only", 0),
            "meteor": model_cache.get("meteor"),
            "bleu": model_cache.get("bleu")
        }

    unique_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = f"outputs/{unique_name}"
    with open(image_path, "wb") as f:
        f.write(content)

    raw_image = Image.open(image_path).convert('RGB')

    # Reuse face analysis if available
    face_info_list = []
    annotated_image_path = ""
    frontend_emotions = []
    age_groups = []
    face_summary = ""

    if "face_summary_en" in cached_entry:
        face_summary = cached_entry["face_summary_en"]
        frontend_emotions = cached_entry["emotions"]
        age_groups = cached_entry["age_groups"]
        annotated_image_path = cached_entry["annotated_image_url"].lstrip("/")
    else:
        try:
            analysis = DeepFace.analyze(img_path=image_path, actions=['age', 'gender', 'emotion'], detector_backend='mtcnn')
            img_cv = cv2.imread(image_path)
            image_width = img_cv.shape[1]
            gender_count = {"Man": 0, "Woman": 0}

            for face in analysis:
                region = face['region']
                emotion = face['dominant_emotion']
                age = int(face['age'])
                gender = face['gender'] if isinstance(face['gender'], str) else max(face['gender'], key=face['gender'].get)
                gender_count[gender] += 1
                label_num = gender_count[gender]
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                age_group = categorize_age_group(age)
                face_info_list.append({
                    "gender": gender,
                    "gender_label": f"{gender.lower()} {label_num}",
                    "age": age,
                    "emotion": emotion,
                    "age_group": age_group,
                    "position": (x, y)
                })
                age_groups.append(f"{gender.lower()} {label_num}: {age_group}")

                label = f"{gender.lower()} {label_num}, {age} yrs, {emotion}, {age_group}"
                cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            annotated_image_path = f"outputs/annotated_{unique_name}"
            cv2.imwrite(annotated_image_path, img_cv)

            frontend_emotions = [f"{f['gender_label']}, {f['age']} yrs, {f['emotion']}, {f['age_group']}" for f in face_info_list]
            face_summary = build_face_summary(face_info_list, image_width)
            image_cache[file_hash] = {
                "annotated_image_url": f"/{annotated_image_path}",
                "languages": {},
                "emotions": frontend_emotions,
                "age_groups": age_groups,
                "face_summary_en": face_summary
            }
        except Exception:
            face_summary = ""
            frontend_emotions = ["Unknown, 0 yrs, Neutral, Unknown"]
            age_groups = ["Unknown"]
            annotated_image_path = image_path
            if file_hash not in image_cache:
                image_cache[file_hash] = {
                    "annotated_image_url": f"/{annotated_image_path}",
                    "languages": {},
                    "emotions": frontend_emotions,
                    "age_groups": age_groups,
                    "face_summary_en": face_summary
                }

    # Always generate caption for selected model
    caption_start = time.time()
    base_caption = generate_caption(raw_image, model_choice).strip()
    caption_end = time.time()
    model_only_time = caption_end - caption_start

    translator = Translator()

    # Translate both full phrases
    try:
        translated_caption = translator.translate(base_caption, dest=lang).text
        print(f"Translated caption: {translated_caption}")
    except:
        translated_caption = base_caption

    try:
        translated_summary = translator.translate(face_summary, dest=lang).text
        translated_summary = convert_numbers_to_local_digits(translated_summary, lang)
        print(f"Translated summary: {translated_summary}")  
    except:
        translated_summary = face_summary

    # Combine
    combined_caption = f"{translated_caption}. {translated_summary}"
    print(f"Combined caption: {combined_caption}")


    #face_summary=face_summary.lower()
    try:
        raw_tts = convert_numbers_to_words(base_caption + ". " + face_summary)
        tts_text = translator.translate(raw_tts, dest=lang).text
    except:
        tts_text = convert_numbers_to_words(base_caption + ". " + face_summary)

    audio_filename = f"{file_hash}_{lang}_{model_choice}.mp3"
    audio_path = f"outputs/{audio_filename}"
    try:
        tts = gTTS(tts_text, lang=lang if lang != 'zh-cn' else 'zh-cn')
        tts.save(audio_path)
    except:
        tts = gTTS(tts_text, lang='en')
        tts.save(audio_path)
    
    meteor_score, bleu_score = evaluate_caption(filename, base_caption)

    
    image_cache[file_hash]["languages"].setdefault(lang, {})[model_choice] = {
        "caption": combined_caption,
        "audio_url": f"/outputs/{audio_filename}",
        "model_time_only": round(model_only_time, 2),
        "meteor": meteor_score,
        "bleu": bleu_score
    }

    return {
        "caption": combined_caption,
        "audio_url": f"/outputs/{audio_filename}",
        "image_url": f"/{annotated_image_path}",
        "emotions": frontend_emotions,
        "age_groups": age_groups,
        "model_time_only": round(model_only_time, 2),
        "meteor": meteor_score,
        "bleu": bleu_score
    }

@app.post("/generate/")
async def generate(file: UploadFile = File(...), lang: str = Form('en'), model: str = Form('blip')):
    content = await file.read()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, process_image, content, file.filename, lang, model)
    return JSONResponse(result)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
