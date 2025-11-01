"""
Image Classification API using Pre-trained Models

Adds vegetable analysis with zero-shot classification (CLIP) to determine
vegetable type and freshness/damage.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
from PIL import Image
import io
import uvicorn
import gc
import os
import requests
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="AI-powered image classification using Google's Vision Transformer",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for the models (lazy-loaded)
classifier = None
zeroshot = None  # Intentionally disabled to save memory on 512MB
ZEROSHOT_API_URL = os.getenv("ZEROSHOT_API_URL", "").strip()

# Supported vegetable labels (extendable)
VEGETABLE_LABELS = [
    "tomato",
    "potato",
    "onion",
    "carrot",
    "cucumber",
    "bell pepper",
    "cabbage",
    "lettuce",
    "eggplant",
    "chili pepper",
    "broccoli",
    "cauliflower",
    "spinach",
    "ginger",
    "garlic",
]

# Simple details database for common vegetables
VEGETABLE_DETAILS = {
    "tomato": {
        "name": "Tomato",
        "description": "A juicy red fruit used as a vegetable in cooking.",
        "fresh_signs": "Firm skin, bright color, fragrant smell.",
        "non_fresh_signs": "Wrinkled skin, soft spots, mold.",
        "storage": "Room temp until ripe, then refrigerate 2–3 days.",
    },
    "potato": {
        "name": "Potato",
        "description": "Starchy tuber, staple ingredient.",
        "fresh_signs": "Firm, dry skin with no green tint.",
        "non_fresh_signs": "Soft, sprouting, green or moldy spots.",
        "storage": "Cool, dark, well-ventilated place (not fridge).",
    },
    "onion": {
        "name": "Onion",
        "description": "Bulb vegetable with pungent flavor.",
        "fresh_signs": "Dry papery skin, firm bulb.",
        "non_fresh_signs": "Soft, wet spots, sprouting.",
        "storage": "Cool, dark, ventilated place (whole).",
    },
    "carrot": {
        "name": "Carrot",
        "description": "Crunchy root vegetable rich in beta-carotene.",
        "fresh_signs": "Firm, vibrant orange, no cracks.",
        "non_fresh_signs": "Limp, black spots, mushy ends.",
        "storage": "Refrigerate in crisper drawer, bagged.",
    },
    "cucumber": {
        "name": "Cucumber",
        "description": "Cool, hydrating gourd often eaten fresh.",
        "fresh_signs": "Firm, glossy skin, no soft spots.",
        "non_fresh_signs": "Yellowing, soft areas, wrinkles.",
        "storage": "Refrigerate; use within a week.",
    },
    "bell pepper": {
        "name": "Bell Pepper",
        "description": "Sweet pepper available in many colors.",
        "fresh_signs": "Taut skin, firm walls, bright color.",
        "non_fresh_signs": "Wrinkling, soft patches, dull color.",
        "storage": "Refrigerate unwashed in crisper.",
    },
    "cabbage": {
        "name": "Cabbage",
        "description": "Leafy head vegetable, green or purple.",
        "fresh_signs": "Tight, heavy head, crisp leaves.",
        "non_fresh_signs": "Loose, wilted leaves, browning.",
        "storage": "Refrigerate wrapped; lasts 1–2 weeks.",
    },
    "lettuce": {
        "name": "Lettuce",
        "description": "Tender leafy greens for salads.",
        "fresh_signs": "Crisp, vibrant leaves.",
        "non_fresh_signs": "Wilted, slimy, browning edges.",
        "storage": "Refrigerate with paper towel in container.",
    },
    "eggplant": {
        "name": "Eggplant",
        "description": "Glossy purple fruit used as a vegetable.",
        "fresh_signs": "Firm, shiny skin, green cap.",
        "non_fresh_signs": "Dull, wrinkled skin, soft spots.",
        "storage": "Cool room or fridge crisper for few days.",
    },
    "chili pepper": {
        "name": "Chili Pepper",
        "description": "Spicy peppers varying in heat.",
        "fresh_signs": "Firm, glossy, taut skin.",
        "non_fresh_signs": "Wrinkled, soft, mold at stem.",
        "storage": "Refrigerate in breathable bag.",
    },
}

@app.on_event("startup")
async def load_model():
    """Startup hook; models are lazy-loaded to reduce memory footprint on 512MB."""
    print("🔄 Starting API (lazy-loading models on first request)...")
    # Do not load any models here to save memory during cold start

@app.get("/")
@app.head("/")
async def root():
    """Serve the custom UI"""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    else:
        # Fallback to API info if HTML not found
        return {
            "message": "Image Classification API",
            "model": "google/mobilenet_v2_1.0_224",
            "status": "running",
            "endpoints": {
                "/classify": "POST - Upload an image to classify",
                "/health": "GET - Check API health status",
                "/docs": "GET - Interactive API documentation"
            }
        }


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Image Classification API",
        "model": "google/mobilenet_v2_1.0_224",
        "status": "running",
        "endpoints": {
            "/classify": "POST - Upload an image to classify",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "zeroshot_loaded": False,
        "vegetable_labels": VEGETABLE_LABELS,
    }


def _resize_image(image: Image.Image, max_side: int = 512) -> Image.Image:
    """Resize image in-place to keep max dimension under max_side to save RAM."""
    try:
        image.thumbnail((max_side, max_side))
    except Exception:
        pass
    return image


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an uploaded image
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with top predictions and confidence scores
    """
    
    # Lazy-load MobileNetV2
    global classifier
    if classifier is None:
        classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles PNG with alpha channel, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to reduce memory footprint
        image = _resize_image(image, max_side=512)
        
        # Classify the image (returns top 5 predictions by default)
        predictions = classifier(image)
        
        # Format response
        response = {
            "success": True,
            "filename": file.filename,
            "predictions": [
                {
                    "label": pred["label"],
                    "confidence": round(pred["score"] * 100, 2),  # Convert to percentage
                    "score": round(pred["score"], 4)
                }
                for pred in predictions
            ],
            "top_prediction": {
                "label": predictions[0]["label"],
                "confidence": round(predictions[0]["score"] * 100, 2)
            }
        }

        # Free memory
        del image, image_bytes, predictions
        gc.collect()

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def _analyze_image(image: Image.Image):
    """Helper to analyze vegetable type and freshness using zero-shot classification."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Vegetable type predictions
    veg_preds = zeroshot(
        image,
        candidate_labels=VEGETABLE_LABELS,
        hypothesis_template="This is a photo of a {}.",
        multi_label=False,
    )

    # Freshness/damage predictions
    freshness_labels = ["fresh", "non-fresh", "damaged"]
    fresh_preds = zeroshot(
        image,
        candidate_labels=freshness_labels,
        hypothesis_template="The vegetable is {}.",
        multi_label=False,
    )

    # Normalize output format
    def _fmt(preds):
        return [
            {
                "label": p["label"],
                "confidence": round(p["score"] * 100, 2),
                "score": round(p["score"], 4),
            }
            for p in preds
        ]

    veg_top = veg_preds[0]["label"] if isinstance(veg_preds, list) and veg_preds else None
    fresh_top = fresh_preds[0]["label"] if isinstance(fresh_preds, list) and fresh_preds else None

    details = VEGETABLE_DETAILS.get(veg_top.lower(), None) if veg_top else None

    return {
        "vegetable": {
            "top": veg_top,
            "predictions": _fmt(veg_preds),
        },
        "freshness": {
            "top": fresh_top,
            "predictions": _fmt(fresh_preds),
        },
        "details": details,
    }


def _run_zeroshot_local(image: Image.Image):
    """Lazily run zero-shot classification locally without persisting the model in memory.
    This loads the CLIP pipeline on-demand, performs predictions, and frees it immediately to save RAM.
    """
    from transformers import pipeline as hf_pipeline

    # Resize to keep memory lower for local ZS
    if image.mode != "RGB":
        image = image.convert("RGB")
    try:
        image.thumbnail((384, 384))
    except Exception:
        pass

    zs = None
    try:
        zs = hf_pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

        veg_preds = zs(
            image,
            candidate_labels=VEGETABLE_LABELS,
            hypothesis_template="This is a photo of a {}.",
            multi_label=False,
        )

        freshness_labels = ["fresh", "slightly wilted", "spoiled", "overripe", "damaged", "non-fresh"]
        fresh_preds = zs(
            image,
            candidate_labels=freshness_labels,
            hypothesis_template="The vegetable is {}.",
            multi_label=False,
        )

        def _fmt(preds):
            return [
                {
                    "label": p["label"],
                    "confidence": round(p["score"] * 100, 2),
                    "score": round(p["score"], 4),
                }
                for p in preds
            ]

        veg_top = veg_preds[0]["label"] if isinstance(veg_preds, list) and veg_preds else None
        fresh_top = fresh_preds[0]["label"] if isinstance(fresh_preds, list) and fresh_preds else None
        details = VEGETABLE_DETAILS.get(veg_top.lower(), None) if veg_top else None

        return {
            "vegetable": {"top": veg_top, "predictions": _fmt(veg_preds)},
            "freshness": {"top": fresh_top, "predictions": _fmt(fresh_preds)},
            "details": details,
        }
    finally:
        # Free model and memory
        del zs
        gc.collect()


def _forward_zeroshot_to_api(file_bytes: bytes, filename: str):
    """Forward zero-shot request to external API if configured.
    Expects external API to accept multipart form field 'file' and return a JSON
    compatible with our /analyze response or close enough to map.
    """
    if not ZEROSHOT_API_URL:
        raise HTTPException(status_code=503, detail="Zero-shot analysis disabled on free tier.")
    try:
        resp = requests.post(
            ZEROSHOT_API_URL,
            files={"file": (filename or "image.jpg", file_bytes, "application/octet-stream")},
            timeout=30,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Upstream zero-shot API error: {resp.text[:200]}")
        data = resp.json()
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed forwarding to zero-shot API: {str(e)}")


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image for vegetable type and freshness/damage using zero-shot classification.
    Returns top predictions and details for the recognized vegetable.
    """
    if zeroshot is None:
        # Offload to external API if configured; otherwise run a local one-shot CLIP and free it
        file_bytes = await file.read()
        if ZEROSHOT_API_URL:
            data = _forward_zeroshot_to_api(file_bytes, file.filename)
            veg_preds = (data.get("vegetable", {}) or {}).get("predictions", [])
            fresh_preds = (data.get("freshness", {}) or {}).get("predictions", [])
        else:
            image = Image.open(io.BytesIO(file_bytes))
            data = _run_zeroshot_local(image)
            veg_preds = data.get("vegetable", {}).get("predictions", [])
            fresh_preds = data.get("freshness", {}).get("predictions", [])

        def _top_and_alt(preds):
            top = preds[0] if preds else {"label": None, "confidence": 0}
            alts = preds[1:4] if len(preds) > 1 else []
            return top, alts
        vtop, valt = _top_and_alt(veg_preds)
        ftop, falt = _top_and_alt(fresh_preds)
        data["summary"] = {
            "top_object": vtop.get("label"),
            "top_object_confidence": vtop.get("confidence"),
            "top_freshness": ftop.get("label"),
            "top_freshness_confidence": ftop.get("confidence"),
            "other_objects": [{"label": p.get("label"), "confidence": p.get("confidence")} for p in valt],
            "other_freshness": [{"label": p.get("label"), "confidence": p.get("confidence")} for p in falt],
        }
        return data

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        analysis = _analyze_image(image)

        # Build structured summary
        veg_preds = analysis.get("vegetable", {}).get("predictions", [])
        fresh_preds = analysis.get("freshness", {}).get("predictions", [])
        def _top_and_alt(preds):
            top = preds[0] if preds else {"label": None, "confidence": 0}
            alts = preds[1:4] if len(preds) > 1 else []
            return top, alts
        vtop, valt = _top_and_alt(veg_preds)
        ftop, falt = _top_and_alt(fresh_preds)

        return {
            "success": True,
            "filename": file.filename,
            **analysis,
            "summary": {
                "top_object": vtop.get("label"),
                "top_object_confidence": vtop.get("confidence"),
                "top_freshness": ftop.get("label"),
                "top_freshness_confidence": ftop.get("confidence"),
                "other_objects": [{"label": p.get("label"), "confidence": p.get("confidence")} for p in valt],
                "other_freshness": [{"label": p.get("label"), "confidence": p.get("confidence")} for p in falt],
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/analyze-top")
async def analyze_image_top_only(file: UploadFile = File(...)):
    """
    Fast analysis endpoint that returns only the top vegetable and top freshness label.
    Useful for real-time webcam polling.
    """
    if zeroshot is None:
        # Offload if configured, else run local one-shot and free
        file_bytes = await file.read()
        if ZEROSHOT_API_URL:
            data = _forward_zeroshot_to_api(file_bytes, file.filename)
        else:
            image = Image.open(io.BytesIO(file_bytes))
            data = _run_zeroshot_local(image)

        veg = None
        fresh = None
        try:
            veg = data.get("vegetable", {}).get("top")
            if not veg and data.get("vegetable"):
                preds = data["vegetable"].get("predictions", [])
                veg = preds[0]["label"] if preds else None
        except Exception:
            pass
        try:
            fresh = data.get("freshness", {}).get("top")
            if not fresh and data.get("freshness"):
                fpreds = data["freshness"].get("predictions", [])
                fresh = fpreds[0]["label"] if fpreds else None
        except Exception:
            pass
        return {
            "success": True,
            "filename": file.filename,
            "vegetable": veg,
            "freshness": fresh,
            "details": data.get("details"),
        }

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        analysis = _analyze_image(image)

        return {
            "success": True,
            "filename": file.filename,
            "vegetable": analysis["vegetable"]["top"],
            "freshness": analysis["freshness"]["top"],
            "details": analysis["details"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/classify-top")
async def classify_image_top_only(file: UploadFile = File(...)):
    """
    Classify an uploaded image and return only the top prediction
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with only the top prediction
    """
    
    # Lazy-load MobileNetV2
    global classifier
    if classifier is None:
        classifier = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to reduce memory footprint
        image = _resize_image(image, max_side=512)

        predictions = classifier(image, top_k=1)  # Get only top prediction
        
        response = {
            "success": True,
            "filename": file.filename,
            "label": predictions[0]["label"],
            "confidence": round(predictions[0]["score"] * 100, 2),
            "score": round(predictions[0]["score"], 4)
        }

        # Free memory
        del image, image_bytes, predictions
        gc.collect()

        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 Image Classification API")
    print("=" * 60)
    print("📦 Model: google/mobilenet_v2_1.0_224")
    print("🌐 Custom UI: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔧 API Info: http://localhost:8000/api")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
