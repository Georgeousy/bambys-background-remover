import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps
from rembg import new_session, remove

app = FastAPI(title="Bamby's Background Remover")

# This is an admin-only helper service. CORS is permissive for easy mobile testing.
# We will secure the Bamby integration separately after the service works.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MODEL_NAME = os.getenv("REMBG_MODEL", "u2netp")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
OUTPUT_SIZE = (1600, 2000)
MARGIN = 80

session = new_session(MODEL_NAME)


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Bamby's Background Remover",
        "model": MODEL_NAME,
    }


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME}


@app.post("/remove")
async def remove_background(image: UploadFile = File(...)):
    raw = await image.read()

    if not raw:
        raise HTTPException(status_code=400, detail="No image was uploaded.")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum is 12 MB.")

    try:
        source = Image.open(io.BytesIO(raw))
        source = ImageOps.exif_transpose(source).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    try:
        foreground = remove(
            source,
            session=session,
            alpha_matting=False,
            post_process_mask=True,
        )

        if not isinstance(foreground, Image.Image):
            foreground = Image.open(io.BytesIO(foreground)).convert("RGBA")
        else:
            foreground = foreground.convert("RGBA")

        # Preserve the whole product. Never crop it.
        max_content = (
            OUTPUT_SIZE[0] - (MARGIN * 2),
            OUTPUT_SIZE[1] - (MARGIN * 2),
        )
        foreground.thumbnail(max_content, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", OUTPUT_SIZE, "white")
        x = (OUTPUT_SIZE[0] - foreground.width) // 2
        y = (OUTPUT_SIZE[1] - foreground.height) // 2

        if foreground.getchannel("A").getbbox():
            canvas.paste(foreground, (x, y), foreground)
        else:
            canvas.paste(foreground.convert("RGB"), (x, y))

        out = io.BytesIO()
        canvas.save(
            out,
            format="JPEG",
            quality=94,
            optimize=True,
            progressive=True,
        )

        return Response(
            content=out.getvalue(),
            media_type="image/jpeg",
            headers={"Content-Disposition": 'inline; filename="bambys-white-background.jpg"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {exc}")
