"""
Brain-Ear Axis Analysis System API Server

This module implements the main API server for the Brain-Ear Axis Analysis System,
providing endpoints for data management, workflow execution, and result retrieval.
"""

import os
import logging
import yaml
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Query, Path as PathParam, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    config_path = os.environ.get('CONFIG_PATH', 'config/config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

try:
    config = load_config()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Brain-Ear Axis Analysis System API",
    description="API for the Brain-Ear Axis Analysis System, providing endpoints for data management, workflow execution, and result retrieval.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api']['cors_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    
class UserInDB(User):
    hashed_password: str
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: Optional[str] = None
    
class PatientData(BaseModel):
    patient_id: str
    age: int
    sex: str
    clinical_history: Dict[str, Any]
    
class AnalysisRequest(BaseModel):
    patient_id: str
    analysis_type: str = Field(..., description="Type of analysis to perform: 'genomic', 'microbiome', 'neural', 'integrated', 'localization', 'personalized'")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class AnalysisStatus(BaseModel):
    job_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None

class DataUploadResponse(BaseModel):
    upload_id: str
    file_name: str
    file_size: int
    status: str
    module: str
    
class AnalysisResult(BaseModel):
    job_id: str
    patient_id: str
    analysis_type: str
    completion_time: datetime
    results: Dict[str, Any]
    
# Authentication functions
def get_user(db, username: str):
    # Mock function - in real implementation, this would query the database
    if username == "testuser":
        return UserInDB(
            username=username,
            email="test@example.com",
            hashed_password="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
            disabled=False
        )

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    # Mock verification - in real implementation, this would use a secure password hash verification
    if password != "password":
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config['api']['jwt_secret'], algorithm="HS256")
    
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, config['api']['jwt_secret'], algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = get_user(None, username=token_data.username)
    if user is None:
        raise credentials_exception
        
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# API Routes
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(None, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(seconds=config['api']['jwt_expiration'])
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Patient data endpoints
@app.post("/patients/", response_model=PatientData)
async def create_patient(
    patient_data: PatientData,
    current_user: User = Depends(get_current_active_user)
):
    # In a real implementation, this would save to the database
    logger.info(f"Creating patient record for {patient_data.patient_id}")
    return patient_data

@app.get("/patients/{patient_id}", response_model=PatientData)
async def get_patient(
    patient_id: str = PathParam(..., description="Unique patient identifier"),
    current_user: User = Depends(get_current_active_user)
):
    # Mock implementation - in real code, this would query the database
    if patient_id == "test123":
        return PatientData(
            patient_id=patient_id,
            age=45,
            sex="M",
            clinical_history={
                "diagnosis": "Tinnitus",
                "onset": "2024-01-15",
                "symptoms": ["ringing in left ear", "occasional dizziness"],
                "medications": ["None"]
            }
        )
    raise HTTPException(status_code=404, detail="Patient not found")

# Data upload endpoints
@app.post("/upload/genomic", response_model=DataUploadResponse)
async def upload_genomic_data(
    patient_id: str = Query(..., description="Patient ID to associate with the upload"),
    data_type: str = Query(..., description="Type of genomic data (e.g., 'wgs', 'wes', 'target_panel')"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    # Mock implementation - in real code, this would save the file to object storage
    upload_id = str(uuid.uuid4())
    logger.info(f"Uploading genomic data for patient {patient_id}, type {data_type}, file {file.filename}")
    
    return DataUploadResponse(
        upload_id=upload_id,
        file_name=file.filename,
        file_size=0,  # In a real implementation, this would be the actual file size
        status="uploaded",
        module="genomic_analysis"
    )

@app.post("/upload/microbiome", response_model=DataUploadResponse)
async def upload_microbiome_data(
    patient_id: str = Query(..., description="Patient ID to associate with the upload"),
    sample_site: str = Query(..., description="Sampling site (e.g., 'ear_canal', 'middle_ear', 'gut')"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    upload_id = str(uuid.uuid4())
    logger.info(f"Uploading microbiome data for patient {patient_id}, site {sample_site}, file {file.filename}")
    
    return DataUploadResponse(
        upload_id=upload_id,
        file_name=file.filename,
        file_size=0,
        status="uploaded",
        module="microbiome_analysis"
    )

@app.post("/upload/neural", response_model=DataUploadResponse)
async def upload_neural_data(
    patient_id: str = Query(..., description="Patient ID to associate with the upload"),
    data_type: str = Query(..., description="Type of neural data (e.g., 'fmri', 'eeg', 'ct')"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    upload_id = str(uuid.uuid4())
    logger.info(f"Uploading neural data for patient {patient_id}, type {data_type}, file {file.filename}")
    
    return DataUploadResponse(
        upload_id=upload_id,
        file_name=file.filename,
        file_size=0,
        status="uploaded",
        module="neural_signal_analysis"
    )

# Analysis endpoints
@app.post("/analysis/", response_model=AnalysisStatus)
async def request_analysis(
    analysis_request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    # Mock implementation - in real code, this would submit a job to the workflow orchestrator
    job_id = str(uuid.uuid4())
    logger.info(f"Starting {analysis_request.analysis_type} analysis for patient {analysis_request.patient_id}")
    
    return AnalysisStatus(
        job_id=job_id,
        status="submitted",
        start_time=datetime.utcnow(),
        progress=0.0,
        message="Analysis job submitted successfully"
    )

@app.get("/analysis/{job_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    job_id: str = PathParam(..., description="Analysis job ID"),
    current_user: User = Depends(get_current_active_user)
):
    # Mock implementation - in real code, this would query the workflow engine
    return AnalysisStatus(
        job_id=job_id,
        status="running",
        start_time=datetime.utcnow() - timedelta(minutes=5),
        progress=0.3,
        message="Processing data..."
    )

@app.get("/analysis/{job_id}/results", response_model=AnalysisResult)
async def get_analysis_results(
    job_id: str = PathParam(..., description="Analysis job ID"),
    current_user: User = Depends(get_current_active_user)
):
    # Mock implementation - in real code, this would retrieve results from the database
    if job_id.startswith("test"):
        return AnalysisResult(
            job_id=job_id,
            patient_id="test123",
            analysis_type="integrated",
            completion_time=datetime.utcnow() - timedelta(minutes=10),
            results={
                "genetic_variants": [
                    {"gene": "KCNE1", "variant": "rs1805127", "pathogenicity": 0.89}
                ],
                "microbiome_findings": {
                    "pseudomonas_aeruginosa_abundance": 0.153,
                    "lactobacillus_abundance": 0.005
                },
                "neural_findings": {
                    "auditory_cortex_connectivity": 0.78,
                    "abnormal_networks": ["auditory-limbic"]
                },
                "integrated_model": {
                    "primary_cause": "genetic_inflammation_pathway",
                    "confidence": 0.82
                }
            }
        )
    raise HTTPException(status_code=404, detail="Results not found or analysis not complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    host = config['api']['host']
    port = config['api']['port']
    debug = config['api']['debug']
    
    logger.info(f"Starting Brain-Ear Axis API server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=debug)
