# download_coco_subset.py
import requests
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(args):
    url, save_path = args
    if os.path.exists(save_path):
        return True
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception:
        return False
    return False

def download_coco_subset(
    save_dir="COCO/images",
    captions_save_path="COCO/captions.csv",
    max_images=20000,
    num_workers=8
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(captions_save_path), exist_ok=True)

    # Download annotations JSON (small file ~241MB for train2017)
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_dir = "COCO/annotations"
    ann_file = os.path.join(ann_dir, "captions_train2017.json")

    if not os.path.exists(ann_file):
        print("Downloading annotations...")
        os.makedirs(ann_dir, exist_ok=True)
        import zipfile, io
        response = requests.get(ann_url, stream=True)
        total = int(response.headers.get('content-length', 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit='B', unit_scale=True, desc="Annotations") as bar:
            for chunk in response.iter_content(chunk_size=8192):
                buf.write(chunk)
                bar.update(len(chunk))
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            z.extractall("COCO")
        print("[INFO] Annotations downloaded and extracted.")

    # Load annotations
    print("Loading annotations...")
    with open(ann_file) as f:
        data = json.load(f)

    # Build id → url and id → filename mapping
    id_to_url  = {img['id']: img['coco_url']   for img in data['images']}
    id_to_file = {img['id']: img['file_name']  for img in data['images']}

    # Group captions by image, limit to max_images
    image_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_captions:
            if len(image_captions) >= max_images:
                continue
            image_captions[img_id] = []
        image_captions[img_id].append(ann['caption'])

    print(f"[INFO] Selected {len(image_captions)} images")

    # Download images in parallel
    download_args = [
        (id_to_url[img_id], os.path.join(save_dir, id_to_file[img_id]))
        for img_id in image_captions
    ]

    print(f"Downloading {len(download_args)} images with {num_workers} workers...")
    success = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_image, arg): arg for arg in download_args}
        with tqdm(total=len(futures), unit="img") as bar:
            for future in as_completed(futures):
                if future.result():
                    success += 1
                bar.update(1)

    print(f"[INFO] Downloaded {success}/{len(download_args)} images")

    # Save captions as CSV (same format as Flickr8k)
    print("Saving captions CSV...")
    with open(captions_save_path, 'w') as f:
        f.write("image,caption\n")
        for img_id, captions in image_captions.items():
            filename = id_to_file[img_id]
            # Only write if image was downloaded
            if os.path.exists(os.path.join(save_dir, filename)):
                for cap in captions:
                    # Escape commas in captions
                    cap = cap.replace('"', '""')
                    f.write(f'{filename},"{cap}"\n')

    print(f"[INFO] Captions saved to {captions_save_path}")


if __name__ == "__main__":
    download_coco_subset(
        save_dir="COCO/images",
        captions_save_path="COCO/captions.csv",
        max_images=20000,
        num_workers=8
    )
