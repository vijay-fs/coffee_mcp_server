from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Use the MongoDB-based implementation for Ragnor document extraction
from routes.ragnor_routes import router as ragnor_router

app = FastAPI()

# Load the table detector model
# model_path = os.path.join(os.path.dirname(
#     __file__), "dynamic_quantized_21.onnx")
# try:
#     table_detector = TableDetector(model_path)
#     print("Table detector model loaded successfully")
# except Exception as e:
#     print(f"Error loading table detector model: {e}")
#     table_detector = None

# table_extractor = TOCRAgent(
#     system_prompt=open('system_prompt.txt', 'r').read())
# diagram_extrator = diagram_extract.DiagramAgent(
#     system_prompt=diagram_extract.system_instruction)
# text_extractor = TextAgent()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:51560", "http://localhost:59343",
                   "http://localhost:3000", "*"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include Ragnor document extraction router
app.include_router(ragnor_router)


@app.middleware("http")
async def add_allow_iframe(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    return response
