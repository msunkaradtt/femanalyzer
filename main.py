import uvicorn
import shutil
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our new service logic
from service import run_physics_analysis

app = FastAPI(title="FEM Physics API")

# Setup CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_physics")
async def analyze_physics_endpoint(file: UploadFile = File(...)):
    """
    Accepts a .glb file, meshes it, runs pressure & thermal sims, 
    and returns deformation vectors.
    """
    # 1. Save uploaded file temporarily
    # We use a temp file so the service can read it from disk
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        input_path = tmp.name

    try:
        # 2. Call the service layer
        result = run_physics_analysis(input_path)
        
        # 3. Return success response
        return {
            "status": "success",
            **result
        }

    except Exception as e:
        # Log error to console for debugging
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Simulation failed: {str(e)}"
        )
        
    finally:
        # 4. Cleanup
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    # Run specifically on port 8001 to avoid conflict with GeoRoughness (8000)
    uvicorn.run(app, host="0.0.0.0", port=8001)