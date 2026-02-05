from __future__ import annotations

import hashlib
import io

import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


API_URL_DEFAULT = "http://localhost:8000/search"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def call_search(api_base: str, image_bytes: bytes, filename: str, content_type: str, top_k: int, top_k_candidates: int, use_crop: bool, crop_x: int, crop_y: int, crop_w: int, crop_h: int) -> dict:
    """Call the search API and return the response."""
    files = {"file": (filename, image_bytes, content_type)}
    params = {"top_k": top_k, "top_k_candidates": top_k_candidates}
    if use_crop:
        params.update({"crop_x": int(crop_x), "crop_y": int(crop_y), "crop_w": int(crop_w), "crop_h": int(crop_h)})
    
    search_url = api_base.rstrip("/") + "/search"
    r = requests.post(search_url, files=files, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def record_click(api_base: str, search_id: str, query_sha256: str, image_id: str, position: int):
    """Record a click event via the API."""
    payload = {
        "search_id": search_id,
        "query_sha256": query_sha256,
        "image_id": image_id,
        "position": position,
    }
    click_url = api_base.rstrip("/") + "/click"
    try:
        r = requests.post(click_url, json=payload, timeout=10)
        r.raise_for_status()
        st.toast(f"Click recorded (pos={position})")
    except Exception as e:
        st.warning(f"Click logging failed: {e}")


def draw_crop_preview(img: Image.Image, crop_x: int, crop_y: int, crop_w: int, crop_h: int) -> Image.Image:
    """Draw a rectangle overlay showing the crop region on the image."""
    img_copy = img.convert("RGBA").copy()
    
    # Draw semi-transparent overlay (darken everything outside crop)
    overlay = Image.new("RGBA", img_copy.size, (0, 0, 0, 128))
    mask = Image.new("L", img_copy.size, 255)
    mask_draw = ImageDraw.Draw(mask)
    # Fill the crop region with transparent (0), keep outside dark
    mask_draw.rectangle([crop_x, crop_y, crop_x + crop_w, crop_y + crop_h], fill=0)
    overlay.putalpha(mask)
    img_copy = Image.alpha_composite(img_copy, overlay)
    
    # Draw crop rectangle border
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([crop_x, crop_y, crop_x + crop_w, crop_y + crop_h], outline="red", width=3)
    
    # Draw corner markers
    corner_size = 10
    corners = [
        (crop_x, crop_y),  # top-left
        (crop_x + crop_w, crop_y),  # top-right
        (crop_x, crop_y + crop_h),  # bottom-left
        (crop_x + crop_w, crop_y + crop_h),  # bottom-right
    ]
    for cx, cy in corners:
        draw.ellipse([cx - corner_size, cy - corner_size, cx + corner_size, cy + corner_size], 
                    fill="red", outline="white", width=2)
    
    return img_copy.convert("RGB")


def main() -> None:
    st.set_page_config(page_title="Visual Search POC", layout="wide")
    st.title("Visual Search System POC")

    # --- Initialize session state ---
    if "results" not in st.session_state:
        st.session_state.results = None
    if "search_id" not in st.session_state:
        st.session_state.search_id = None
    if "query_sha256" not in st.session_state:
        st.session_state.query_sha256 = None
    if "query_bytes" not in st.session_state:
        st.session_state.query_bytes = None
    if "query_filename" not in st.session_state:
        st.session_state.query_filename = None
    if "query_content_type" not in st.session_state:
        st.session_state.query_content_type = None
    if "selected_uri" not in st.session_state:
        st.session_state.selected_uri = None
    if "selected_score" not in st.session_state:
        st.session_state.selected_score = None
    if "system_versions" not in st.session_state:
        st.session_state.system_versions = None
    if "crop_x" not in st.session_state:
        st.session_state.crop_x = 0
    if "crop_y" not in st.session_state:
        st.session_state.crop_y = 0
    if "crop_w" not in st.session_state:
        st.session_state.crop_w = 224
    if "crop_h" not in st.session_state:
        st.session_state.crop_h = 224
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "crop_mode" not in st.session_state:
        st.session_state.crop_mode = "sliders"  # "sliders" or "mouse"
    if "crop_click1" not in st.session_state:
        st.session_state.crop_click1 = None  # First click (top-left)
    if "crop_click2" not in st.session_state:
        st.session_state.crop_click2 = None  # Second click (bottom-right)

    st.sidebar.header("Search Settings")
    api_url_input = st.sidebar.text_input("API URL", value=API_URL_DEFAULT)
    # Extract base URL (remove /search if present)
    api_base = api_url_input.replace("/search", "").rstrip("/")
    top_k = st.sidebar.number_input("Top-K", min_value=1, max_value=200, value=50)
    top_k_candidates = st.sidebar.number_input("Top-K Candidates (ANN)", min_value=10, max_value=2000, value=200)

    uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])
    
    # Handle uploaded image and update session state
    if uploaded is not None:
        uploaded_img = Image.open(uploaded).convert("RGB")
        st.session_state.uploaded_image = uploaded_img
        
        # Reset crop to image center if dimensions changed
        img_w, img_h = uploaded_img.size
        if st.session_state.crop_x >= img_w or st.session_state.crop_y >= img_h:
            st.session_state.crop_x = max(0, (img_w - st.session_state.crop_w) // 2)
            st.session_state.crop_y = max(0, (img_h - st.session_state.crop_h) // 2)
        
        # Ensure crop doesn't exceed image bounds
        st.session_state.crop_w = min(st.session_state.crop_w, img_w)
        st.session_state.crop_h = min(st.session_state.crop_h, img_h)
        st.session_state.crop_x = min(st.session_state.crop_x, img_w - st.session_state.crop_w)
        st.session_state.crop_y = min(st.session_state.crop_y, img_h - st.session_state.crop_h)
    else:
        st.session_state.uploaded_image = None

    # Crop controls (only show if image is uploaded)
    if st.session_state.uploaded_image is not None:
        img_w, img_h = st.session_state.uploaded_image.size
        
        st.sidebar.header("Optional Crop (Milestone C)")
        use_crop = st.sidebar.checkbox("Use crop box", value=False)
        
        if use_crop:
            st.sidebar.caption(f"Image size: {img_w} × {img_h}")
            
            # Crop mode selector
            crop_mode = st.sidebar.radio(
                "Crop method:",
                ["sliders", "mouse"],
                index=0 if st.session_state.crop_mode == "sliders" else 1,
                key="crop_mode_radio"
            )
            st.session_state.crop_mode = crop_mode
            
            if crop_mode == "sliders":
                # Use sliders for better UX
                crop_x = st.sidebar.slider(
                    "Crop X (left)", 
                    min_value=0, 
                    max_value=max(0, img_w - 1), 
                    value=st.session_state.crop_x,
                    key="crop_x_slider"
                )
                crop_y = st.sidebar.slider(
                    "Crop Y (top)", 
                    min_value=0, 
                    max_value=max(0, img_h - 1), 
                    value=st.session_state.crop_y,
                    key="crop_y_slider"
                )
                crop_w = st.sidebar.slider(
                    "Crop Width", 
                    min_value=1, 
                    max_value=img_w - crop_x, 
                    value=min(st.session_state.crop_w, img_w - crop_x),
                    key="crop_w_slider"
                )
                crop_h = st.sidebar.slider(
                    "Crop Height", 
                    min_value=1, 
                    max_value=img_h - crop_y, 
                    value=min(st.session_state.crop_h, img_h - crop_y),
                    key="crop_h_slider"
                )
                
                # Update session state
                st.session_state.crop_x = crop_x
                st.session_state.crop_y = crop_y
                st.session_state.crop_w = crop_w
                st.session_state.crop_h = crop_h
                
                # Show crop preview
                st.sidebar.caption(f"Crop: ({crop_x}, {crop_y}) {crop_w}×{crop_h}")
            else:  # mouse mode
                st.sidebar.info("👆 Click on the image below to set crop region:\n1. First click = top-left corner\n2. Second click = bottom-right corner")
                
                # Reset button
                if st.sidebar.button("Reset crop selection"):
                    st.session_state.crop_click1 = None
                    st.session_state.crop_click2 = None
                    st.session_state.crop_x = 0
                    st.session_state.crop_y = 0
                    st.session_state.crop_w = 224
                    st.session_state.crop_h = 224
                
                # Display current crop from clicks or use defaults
                if st.session_state.crop_click1 is not None and st.session_state.crop_click2 is not None:
                    # Calculate crop from two clicks
                    x1, y1 = st.session_state.crop_click1
                    x2, y2 = st.session_state.crop_click2
                    
                    # Ensure proper ordering (top-left to bottom-right)
                    crop_x = min(x1, x2)
                    crop_y = min(y1, y2)
                    crop_w = abs(x2 - x1)
                    crop_h = abs(y2 - y1)
                    
                    # Ensure within bounds
                    crop_x = max(0, min(crop_x, img_w - 1))
                    crop_y = max(0, min(crop_y, img_h - 1))
                    crop_w = max(1, min(crop_w, img_w - crop_x))
                    crop_h = max(1, min(crop_h, img_h - crop_y))
                    
                    st.session_state.crop_x = crop_x
                    st.session_state.crop_y = crop_y
                    st.session_state.crop_w = crop_w
                    st.session_state.crop_h = crop_h
                    
                    st.sidebar.caption(f"Crop: ({crop_x}, {crop_y}) {crop_w}×{crop_h}")
                else:
                    # Use existing crop values
                    crop_x = st.session_state.crop_x
                    crop_y = st.session_state.crop_y
                    crop_w = st.session_state.crop_w
                    crop_h = st.session_state.crop_h
                    
                    if st.session_state.crop_click1 is not None:
                        st.sidebar.caption(f"First click: {st.session_state.crop_click1}\n👆 Click again for bottom-right corner")
        else:
            crop_x = st.session_state.crop_x
            crop_y = st.session_state.crop_y
            crop_w = st.session_state.crop_w
            crop_h = st.session_state.crop_h
    else:
        use_crop = False
        crop_x = 0
        crop_y = 0
        crop_w = 224
        crop_h = 224
    
    # Search button - only trigger search when clicked
    colA, colB, colC = st.columns([1, 1, 1])
    run = colA.button("Search", type="primary", disabled=(uploaded is None))

    if run and uploaded is not None:
        q_bytes = uploaded.getvalue()
        st.session_state.query_bytes = q_bytes
        st.session_state.query_sha256 = sha256_bytes(q_bytes)
        st.session_state.query_filename = uploaded.name
        st.session_state.query_content_type = uploaded.type or "application/octet-stream"

        with st.spinner("Searching..."):
            try:
                resp = call_search(
                    api_base=api_base,
                    image_bytes=q_bytes,
                    filename=st.session_state.query_filename,
                    content_type=st.session_state.query_content_type,
                    top_k=int(top_k),
                    top_k_candidates=int(top_k_candidates),
                    use_crop=use_crop,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
                st.session_state.search_id = resp.get("search_id")
                st.session_state.results = resp.get("results", [])
                st.session_state.system_versions = {
                    "model_version": resp.get("model_version"),
                    "preprocess_version": resp.get("preprocess_version"),
                    "index_version": resp.get("index_version"),
                }
                st.session_state.selected_uri = None
                st.session_state.selected_score = None
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.results = None

    # Display uploaded image with crop preview (before search)
    if st.session_state.uploaded_image is not None:
        st.subheader("Query Image")
        img = st.session_state.uploaded_image
        img_w, img_h = img.size
        
        if use_crop:
            if st.session_state.crop_mode == "mouse":
                # Interactive image with click-to-select
                st.caption("👆 Click on the image below to set crop region (first click = top-left, second click = bottom-right)")
                
                # Prepare image with crop preview and click markers for display
                display_img = img.copy()
                if st.session_state.crop_click1 is not None or st.session_state.crop_click2 is not None:
                    # Draw crop preview overlay
                    display_img = draw_crop_preview(img, crop_x, crop_y, crop_w, crop_h)
                    
                    # Draw click markers on top
                    draw = ImageDraw.Draw(display_img.convert("RGBA"))
                    if st.session_state.crop_click1 is not None:
                        cx, cy = st.session_state.crop_click1
                        # Draw a larger, more visible marker
                        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill="blue", outline="white", width=3)
                    if st.session_state.crop_click2 is not None:
                        cx, cy = st.session_state.crop_click2
                        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill="green", outline="white", width=3)
                    display_img = display_img.convert("RGB")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Click on image to set crop")
                    # Get click coordinates - this creates and displays the clickable image
                    # streamlit_image_coordinates accepts PIL Image, numpy array, or file path
                    # Pass the PIL Image directly
                    click_coords = streamlit_image_coordinates(
                        display_img,
                        key="crop_image_clicker",
                        width=img_w,
                    )
                    
                    # Handle click
                    if click_coords is not None:
                        click_x = click_coords["x"]
                        click_y = click_coords["y"]
                        
                        # Clamp to image bounds (coordinates should already be in original image space)
                        click_x = max(0, min(click_x, img_w - 1))
                        click_y = max(0, min(click_y, img_h - 1))
                        
                        if st.session_state.crop_click1 is None:
                            # First click - set top-left
                            st.session_state.crop_click1 = (click_x, click_y)
                            st.session_state.crop_click2 = None
                            st.rerun()
                        elif st.session_state.crop_click2 is None:
                            # Second click - set bottom-right
                            st.session_state.crop_click2 = (click_x, click_y)
                            st.rerun()
                
                with col2:
                    st.caption("Cropped preview")
                    try:
                        cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                        st.image(cropped, caption=f"Crop: ({crop_x}, {crop_y}) {crop_w}×{crop_h}", use_container_width=True)
                    except Exception as e:
                        st.error(f"Crop preview error: {e}")
            else:
                # Slider mode - show image with crop overlay
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Original with crop preview")
                    preview_img = draw_crop_preview(img, crop_x, crop_y, crop_w, crop_h)
                    st.image(preview_img, caption=f"Crop: ({crop_x}, {crop_y}) {crop_w}×{crop_h}", use_container_width=True)
                
                with col2:
                    st.caption("Cropped preview")
                    try:
                        cropped = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                        st.image(cropped, caption=f"Cropped region", use_container_width=True)
                    except Exception as e:
                        st.error(f"Crop preview error: {e}")
        else:
            st.image(img, caption="Query Image", use_container_width=True)
    
    # Display query image from session state (after search)
    if st.session_state.query_bytes is not None:
        st.subheader("Query Used for Search")
        img = Image.open(io.BytesIO(st.session_state.query_bytes)).convert("RGB")
        st.image(img, caption="Query Image", use_container_width=True)

    # Display selected preview
    if st.session_state.selected_uri:
        st.subheader("Selected")
        try:
            selected_img = Image.open(st.session_state.selected_uri).convert("RGB")
            st.image(selected_img, use_container_width=True)
        except Exception:
            st.write(st.session_state.selected_uri)
        if st.session_state.selected_score is not None:
            st.caption(f"Similarity: {st.session_state.selected_score:.4f}")

    # Display system versions
    if st.session_state.system_versions:
        st.subheader("System Versions")
        st.write(st.session_state.system_versions)

    # Display results from session state
    results = st.session_state.results
    if results:
        st.subheader(f"Results (Top {len(results)})")
        cols = st.columns(5)

        for i, item in enumerate(results, start=1):
            c = cols[(i - 1) % 5]
            with c:
                st.caption(f"pos={item['position']} score={item['score']:.3f}")
                st.caption(item["image_id"][:12] + "…")
                try:
                    ri = Image.open(item["uri"]).convert("RGB")
                    st.image(ri, use_container_width=True)
                except Exception:
                    st.write(item["uri"])

                # Record click without opening anything - use closure to capture item and position
                def make_click_callback(_item, _i):
                    def click_callback():
                        record_click(
                            api_base,
                            st.session_state.search_id,
                            st.session_state.query_sha256,
                            _item["image_id"],
                            _i,
                        )
                        st.session_state.selected_uri = _item["uri"]
                        st.session_state.selected_score = _item.get("score")
                    return click_callback

                st.button(
                    "Select",
                    key=f"select_{item['image_id']}_{i}",
                    on_click=make_click_callback(item, i),
                )


if __name__ == "__main__":
    main()
