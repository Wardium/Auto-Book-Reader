import os
import time
import math
import threading
import pyautogui
import pytesseract
import cv2
import numpy as np
import imagehash
from PIL import Image, ImageChops, ImageStat
from difflib import SequenceMatcher
from pynput import keyboard
from fpdf import FPDF

# --- CONFIGURATION ---
OUTPUT_PDF = "book.pdf"
NEXT_BTN_IMG = "next_button.png"
IMG_SAVE_DIR = "extracted_images"

# Visual settings
MAX_BTN_MOVEMENT = 300  # Max distance (in points) the button can be from your original hover

# State variables
STOP_PHRASE = "" 
is_scraping = False
seen_image_hashes = []
book_pages = []  
space_times = []
total_views_processed = 0  
original_btn_coords = (0, 0) 

# Boundary Setup Variables
setup_step = 1
crop_top_left = None
crop_bottom_right = None

if not os.path.exists(IMG_SAVE_DIR):
    os.makedirs(IMG_SAVE_DIR)

def get_retina_scale():
    screen_width_points = pyautogui.size()[0]
    screenshot_width_pixels = pyautogui.screenshot().width
    return screenshot_width_pixels / screen_width_points

def capture_reference_image():
    global original_btn_coords
    mx, my = pyautogui.position()
    original_btn_coords = (mx, my) 
    
    scale = get_retina_scale()
    px_mx, px_my = mx * scale, my * scale
    w, h = 120 * scale, 60 * scale
    left, top = int(px_mx - (w / 2)), int(px_my - (h / 2))
    
    print(f"\n[+] Capturing button at Mouse({int(mx)}, {int(my)})...")
    img = pyautogui.screenshot()
    cropped = img.crop((left, top, left + w, top + h))
    
    cropped.convert('RGB').save(NEXT_BTN_IMG, icc_profile=None) 
    print(f"[✓] Saved reference image: '{NEXT_BTN_IMG}'.")

def is_stop_phrase_valid(text, phrase):
    phrase = phrase.lower().strip()
    lines = [line.strip().lower() for line in text.split('\n') if line.strip()]
    
    if not lines:
        return False
        
    for line in lines:
        cleaned_line = line.replace('.', '').replace(':', '').replace('-', '').strip()
        if cleaned_line == phrase:
            return True
            
    if lines[0].startswith(phrase):
        return True
        
    return False

def merge_text(existing_text, new_text):
    if not existing_text: return new_text
    tail, head = existing_text[-600:], new_text[:600]
    match = SequenceMatcher(None, tail, head).find_longest_match(0, len(tail), 0, len(head))
    if match.size > 15:
        return existing_text + new_text[match.b + match.size:]
    return existing_text + "\n\n" + new_text

def extract_images_from_screen(pil_img):
    open_cv_image = np.array(pil_img)
    img_bgr = open_cv_image[:, :, ::-1].copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    new_images = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 200 and h > 200 and w < pil_img.width * 0.9:
            cropped = pil_img.crop((x, y, x+w, y+h))
            img_hash = imagehash.average_hash(cropped)
            
            is_duplicate = False
            for seen_hash in seen_image_hashes:
                if img_hash - seen_hash < 5: 
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                seen_image_hashes.append(img_hash)
                img_path = os.path.join(IMG_SAVE_DIR, f"img_{len(seen_image_hashes)}.png")
                cropped.save(img_path)
                new_images.append(img_path)
                
    return new_images

def build_pdf():
    print(f"\nBuilding final PDF: {OUTPUT_PDF}...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", size=11) 
    
    for page_idx, page_content in enumerate(book_pages):
        pdf.add_page() 
        
        for item_type, content in page_content:
            if item_type == "text":
                safe_text = content.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 5, safe_text)
                pdf.ln(5)
            elif item_type == "image":
                pdf.ln(5)
                pdf.image(content, x=None, y=None, w=180)
                pdf.ln(5)
                
    pdf.output(OUTPUT_PDF)
    print(f"[✓] Success! PDF saved to {OUTPUT_PDF}")

def start_scraping_loop():
    global is_scraping, book_pages, total_views_processed
    print("\n--- Scraping Started ---")
    time.sleep(1.0) 
    
    page_count = 1
    screen_w, screen_h = pyautogui.size()
    scale = get_retina_scale()
    
    # Calculate the exact pixel boundaries from the user's setup phase
    box_left = int(min(crop_top_left[0], crop_bottom_right[0]) * scale)
    box_top = int(min(crop_top_left[1], crop_bottom_right[1]) * scale)
    box_right = int(max(crop_top_left[0], crop_bottom_right[0]) * scale)
    box_bottom = int(max(crop_top_left[1], crop_bottom_right[1]) * scale)
    
    try:
        while is_scraping:
            print(f"\n--- Processing Site Page {page_count} ---")
            
            current_page_memory = []
            pyautogui.click(screen_w / 2, screen_h / 2) 
            time.sleep(0.5)
            
            section_text = ""
            last_extracted_text = ""
            
            while True:
                total_views_processed += 1
                raw_screenshot = pyautogui.screenshot()
                
                # Use the custom bounding box
                img = raw_screenshot.crop((box_left, box_top, box_right, box_bottom))
                
                gray_img = img.convert('L')
                current_text = pytesseract.image_to_string(gray_img).strip()
                
                if STOP_PHRASE and total_views_processed > 2:
                    if is_stop_phrase_valid(current_text, STOP_PHRASE):
                        print(f"\n[!] Stop phrase '{STOP_PHRASE}' found exactly as requested. Stopping.")
                        section_text = merge_text(section_text, current_text)
                        is_scraping = False
                        break
                
                if current_text == last_extracted_text and last_extracted_text != "":
                    print("  -> Bottom of site page reached.")
                    break 
                    
                section_text = merge_text(section_text, current_text)
                last_extracted_text = current_text
                
                found_images = extract_images_from_screen(img)
                for img_path in found_images:
                    print(f"  -> Extracted graphic: {img_path}")
                    current_page_memory.append(("text", section_text))
                    current_page_memory.append(("image", img_path))
                    section_text = "" 
                
                if not is_scraping: 
                    break

                print("  -> Scrolling down...")
                pyautogui.press('pagedown')
                time.sleep(1.2)
            
            if section_text:
                current_page_memory.append(("text", section_text))
                
            book_pages.append(current_page_memory)
            
            if not is_scraping: break
            
            print(f"  -> Searching for '{NEXT_BTN_IMG}'...")
            
            try:
                matches = list(pyautogui.locateAllOnScreen(NEXT_BTN_IMG, confidence=0.85, grayscale=True))
            except Exception:
                matches = []
            
            if matches:
                best_match = None
                min_distance = float('inf')
                
                for match in matches:
                    center_x = (match.left + (match.width / 2)) / scale
                    center_y = (match.top + (match.height / 2)) / scale
                    
                    dist = math.hypot(center_x - original_btn_coords[0], center_y - original_btn_coords[1])
                    
                    if dist < min_distance:
                        min_distance = dist
                        best_match = (center_x, center_y)
                
                if best_match and min_distance <= MAX_BTN_MOVEMENT:
                    print(f"  -> Target found {int(min_distance)} points away. Clicking.")
                    pyautogui.moveTo(best_match[0], best_match[1], duration=0.2)
                    pyautogui.click()
                else:
                    if best_match:
                        print(f"\n[!] Found a button, but it was {int(min_distance)} points away (exceeds {MAX_BTN_MOVEMENT} limit). Ignoring it.")
                    print(f"[!] Could not find '{NEXT_BTN_IMG}' in the valid area. Stopping.")
                    break
            else:
                print(f"\n[!] Could not find '{NEXT_BTN_IMG}'. Stopping.")
                break
                
            time.sleep(2.5) 
            page_count += 1

    except Exception as e:
        print(f"Error occurred: {repr(e)}") 
    finally:
        build_pdf()
        is_scraping = False
        print("\n[✓] Program finished successfully. Exiting...")
        os._exit(0)  # Instantly terminates the script and all background listeners

# --- KEYBOARD LISTENER ---
def on_key_press(key):
    global space_times, is_scraping, setup_step, crop_top_left, crop_bottom_right
    
    if is_scraping: return

    # Intercept '1' and '2' keys for the setup phase
    try:
        if hasattr(key, 'char'):
            if key.char == '1' and setup_step == 1:
                crop_top_left = pyautogui.position()
                print(f"[✓] Top-Left corner locked at {crop_top_left}.")
                print("\n-> STEP 2: Hover over the BOTTOM-RIGHT corner of the text and press '2'.")
                setup_step = 2
                return
            elif key.char == '2' and setup_step == 2:
                crop_bottom_right = pyautogui.position()
                print(f"[✓] Bottom-Right corner locked at {crop_bottom_right}.")
                print("\n-> STEP 3: Hover directly over the 'Next Page' button and DOUBLE-TAP SPACE to start.")
                setup_step = 3
                return
    except Exception:
        pass

    # Double-tap Spacebar listener (Only unlocks after steps 1 & 2)
    if key == keyboard.Key.space and setup_step == 3:
        now = time.time()
        space_times.append(now)
        
        if len(space_times) > 2:
            space_times.pop(0)
            
        if len(space_times) == 2 and (space_times[1] - space_times[0]) < 0.6:
            is_scraping = True
            space_times = [] 
            
            capture_reference_image()
            threading.Thread(target=start_scraping_loop).start()

if __name__ == "__main__":
    print("--- Pro Auto-Scraper Ready ---")
    
    STOP_PHRASE = input("Enter a Stop Phrase (e.g., 'Chapter 2') or press Enter to skip: ").strip()
    
    if STOP_PHRASE:
        print(f"\n[!] Script will automatically stop when it reads: '{STOP_PHRASE}'")
    else:
        print("\n[!] No stop phrase entered. Script will run until stopped manually or no Next button is found.")

    print("\n-----------------------------------")
    print("MANUAL BOUNDARY SETUP:")
    print("STEP 1: Hover your mouse over the TOP-LEFT corner of the reading area and press '1'.")
    print("-----------------------------------")
    
    with keyboard.Listener(on_press=on_key_press) as kb_listener:
        kb_listener.join()
