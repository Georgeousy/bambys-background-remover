import io
import os
import gc
import threading

# Keep native numerical libraries conservative on Render's 512 MB free instance.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps
from rembg import new_session, remove

app = FastAPI(title="Bamby's Background Remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("REMBG_MODEL", "u2netp")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
AI_MAX_SIDE = 960
OUTPUT_SIZE = (1600, 2000)
MARGIN = 80

# IMPORTANT:
# Do NOT create the rembg/ONNX session while the web server is starting.
# Render needs the web port to open quickly. We create the model lazily on the
# first /remove request instead.
SESSION = None
SESSION_LOCK = threading.Lock()


def get_session():
    global SESSION
    if SESSION is None:
        with SESSION_LOCK:
            if SESSION is None:
                SESSION = new_session(
                    MODEL_NAME,
                    providers=["CPUExecutionProvider"],
                )
    return SESSION


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Bamby's Background Remover",
        "model": MODEL_NAME,
        "mode": "lazy-low-memory",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_NAME,
        "mode": "lazy-low-memory",
        "model_loaded": SESSION is not None,
    }


@app.post("/remove")
async def remove_background(image: UploadFile = File(...)):
    raw = await image.read()

    if not raw:
        raise HTTPException(status_code=400, detail="No image was uploaded.")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum is 12 MB.")

    try:
        original = Image.open(io.BytesIO(raw))
        original = ImageOps.exif_transpose(original).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        working = original.copy()
        working.thumbnail((AI_MAX_SIDE, AI_MAX_SIDE), Image.Resampling.LANCZOS)

        # Load the AI model only when background removal is actually requested.
        session = get_session()

        foreground = remove(
            working,
            session=session,
            alpha_matting=False,
            post_process_mask=False,
        )

        if not isinstance(foreground, Image.Image):
            foreground = Image.open(io.BytesIO(foreground)).convert("RGBA")
        else:
            foreground = foreground.convert("RGBA")

        original.close()
        del original, working, raw
        gc.collect()

        max_content = (
            OUTPUT_SIZE[0] - (MARGIN * 2),
            OUTPUT_SIZE[1] - (MARGIN * 2),
        )
        foreground.thumbnail(max_content, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", OUTPUT_SIZE, "white")
        x = (OUTPUT_SIZE[0] - foreground.width) // 2
        y = (OUTPUT_SIZE[1] - foreground.height) // 2
        canvas.paste(foreground, (x, y), foreground)

        out = io.BytesIO()
        canvas.save(
            out,
            format="JPEG",
            quality=94,
            optimize=False,
            progressive=False,
        )

        result = out.getvalue()

        foreground.close()
        canvas.close()
        out.close()
        gc.collect()

        return Response(
            content=result,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": 'inline; filename="bambys-white-background.jpg"',
                "Cache-Control": "no-store",
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}")
