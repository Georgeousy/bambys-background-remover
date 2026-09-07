# Bamby's Background Remover

Small online image-processing API for Bamby's Collection.

## Endpoints

- `GET /health` — confirms that the service is running.
- `POST /remove` — upload one image as multipart form-data using the field name `image`.

The service removes the background, preserves the complete product, places it on a white 1600 x 2000 canvas, and returns a high-quality JPEG.

## Render settings

Build command:

    pip install -r requirements.txt

Start command:

    uvicorn server:app --host 0.0.0.0 --port $PORT

Environment variable:

    REMBG_MODEL=u2netp

`u2netp` is intentionally used for the first free Render deployment because it is much smaller than the full U2Net model and is more suitable for a low-memory free instance.

## Important

This first deployment is only for testing the online processor. Do not connect it to the production Bamby app until the `/health` and `/remove` tests both work.
