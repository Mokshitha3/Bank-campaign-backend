from fastapi import FastAPI, APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import requests
import bcrypt
from jose import jwt, JWTError
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
SECRET_KEY = os.environ.get('JWT_SECRET', 'bank-marketing-secret-key-2024')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer()

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Models
class UserRegister(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    password_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

class AnalyticsResponse(BaseModel):
    total_campaigns: int
    success_rate: float
    conversion_rate: float
    avg_age: float
    avg_balance: float
    avg_duration: float

class FilterRequest(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    jobs: Optional[List[str]] = None
    months: Optional[List[str]] = None

# ML Model global variables
ml_model = None
scaler = None
label_encoders = {}

# Helper functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError as e:
        logger.error(f"JWT Error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(status_code=401, detail="Authentication failed")

async def load_bank_data():
    """Load UCI Bank Marketing dataset"""
    try:
        # Check if data already exists
        count = await db.campaigns.count_documents({})
        if count > 0:
            logger.info(f"Data already loaded: {count} records")
            return
        
        # Download dataset from UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
        logger.info("Downloading UCI Bank Marketing dataset...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract and read CSV from zip
        import zipfile
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('bank.csv') as f:
                df = pd.read_csv(f, sep=';')
        
        logger.info(f"Dataset loaded: {len(df)} records")
        
        # Convert to dict and insert into MongoDB
        records = df.to_dict('records')
        await db.campaigns.insert_many(records)
        
        logger.info("Dataset successfully stored in MongoDB")
        
    except Exception as e:
        logger.error(f"Error loading bank data: {str(e)}")
        # If download fails, create sample data
        sample_data = [
            {"age": 30, "job": "technician", "marital": "single", "education": "secondary", 
             "default": "no", "balance": 1500, "housing": "yes", "loan": "no", 
             "contact": "cellular", "day": 15, "month": "may", "duration": 250, 
             "campaign": 2, "pdays": -1, "previous": 0, "poutcome": "unknown", "y": "yes"},
            {"age": 45, "job": "management", "marital": "married", "education": "tertiary", 
             "default": "no", "balance": 3000, "housing": "yes", "loan": "yes", 
             "contact": "cellular", "day": 20, "month": "jun", "duration": 180, 
             "campaign": 1, "pdays": -1, "previous": 0, "poutcome": "unknown", "y": "no"}
        ]
        await db.campaigns.insert_many(sample_data * 100)
        logger.info("Sample data inserted")

async def train_ml_model():
    """Train logistic regression model for predictions"""
    global ml_model, scaler, label_encoders
    
    try:
        # Fetch all campaign data
        campaigns = await db.campaigns.find({}, {"_id": 0}).to_list(10000)
        
        if not campaigns:
            logger.warning("No data available for training")
            return
        
        df = pd.DataFrame(campaigns)
        
        # Select features
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        
        # Encode categorical variables
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].astype(str))
            label_encoders[feature] = le
        
        # Prepare features and target
        X = df[categorical_features + numerical_features]
        y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
        
        # Scale numerical features
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        
        # Train model
        ml_model = LogisticRegression(max_iter=1000, random_state=42)
        ml_model.fit(X, y)
        
        logger.info(f"Model trained successfully. Accuracy: {ml_model.score(X, y):.2f}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")

# Auth endpoints
@api_router.post("/auth/register", response_model=TokenResponse)
async def register(user_data: UserRegister):
    # Check if user exists
    existing = await db.users.find_one({"username": user_data.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Hash password
    password_hash = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Create user
    user = User(
        username=user_data.username,
        password_hash=password_hash
    )
    
    user_dict = user.model_dump()
    user_dict['created_at'] = user_dict['created_at'].isoformat()
    await db.users.insert_one(user_dict)
    
    # Create token
    access_token = create_access_token(data={"sub": user.username})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        username=user.username
    )

@api_router.post("/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    # Find user
    user = await db.users.find_one({"username": user_data.username})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not bcrypt.checkpw(user_data.password.encode('utf-8'), user['password_hash'].encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    access_token = create_access_token(data={"sub": user['username']})
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        username=user['username']
    )

# Dashboard endpoints
@api_router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(current_user: str = Depends(get_current_user)):
    campaigns = await db.campaigns.find({}, {"_id": 0}).to_list(10000)
    
    if not campaigns:
        return AnalyticsResponse(
            total_campaigns=0,
            success_rate=0.0,
            conversion_rate=0.0,
            avg_age=0.0,
            avg_balance=0.0,
            avg_duration=0.0
        )
    
    df = pd.DataFrame(campaigns)
    
    total = len(df)
    success_count = len(df[df['y'] == 'yes'])
    success_rate = (success_count / total * 100) if total > 0 else 0
    
    return AnalyticsResponse(
        total_campaigns=total,
        success_rate=round(success_rate, 2),
        conversion_rate=round(success_rate, 2),
        avg_age=round(df['age'].mean(), 1),
        avg_balance=round(df['balance'].mean(), 2),
        avg_duration=round(df['duration'].mean(), 2)
    )

@api_router.post("/analytics/filter")
async def get_filtered_data(filter_req: FilterRequest, current_user: str = Depends(get_current_user)):
    query = {}
    
    if filter_req.age_min is not None or filter_req.age_max is not None:
        query['age'] = {}
        if filter_req.age_min is not None:
            query['age']['$gte'] = filter_req.age_min
        if filter_req.age_max is not None:
            query['age']['$lte'] = filter_req.age_max
    
    if filter_req.jobs:
        query['job'] = {'$in': filter_req.jobs}
    
    if filter_req.months:
        query['month'] = {'$in': filter_req.months}
    
    campaigns = await db.campaigns.find(query, {"_id": 0}).to_list(10000)
    return campaigns

@api_router.get("/charts/age-distribution")
async def get_age_distribution(current_user: str = Depends(get_current_user)):
    campaigns = await db.campaigns.find({}, {"_id": 0, "age": 1, "y": 1}).to_list(10000)
    df = pd.DataFrame(campaigns)
    
    # Create age bins
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    result = df.groupby('age_group').size().reset_index(name='count')
    return result.to_dict('records')

@api_router.get("/charts/job-distribution")
async def get_job_distribution(current_user: str = Depends(get_current_user)):
    campaigns = await db.campaigns.find({}, {"_id": 0, "job": 1}).to_list(10000)
    df = pd.DataFrame(campaigns)
    
    result = df['job'].value_counts().reset_index()
    result.columns = ['job', 'count']
    return result.to_dict('records')

@api_router.get("/charts/monthly-performance")
async def get_monthly_performance(current_user: str = Depends(get_current_user)):
    campaigns = await db.campaigns.find({}, {"_id": 0, "month": 1, "y": 1}).to_list(10000)
    df = pd.DataFrame(campaigns)
    
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    result = df.groupby(['month', 'y']).size().reset_index(name='count')
    result['month'] = pd.Categorical(result['month'], categories=month_order, ordered=True)
    result = result.sort_values('month')
    
    # Pivot to get success and failure counts
    pivot = result.pivot_table(index='month', columns='y', values='count', fill_value=0).reset_index()
    pivot.columns.name = None
    
    return pivot.to_dict('records')

@api_router.get("/charts/balance-duration")
async def get_balance_duration(current_user: str = Depends(get_current_user)):
    campaigns = await db.campaigns.find({}, {"_id": 0, "balance": 1, "duration": 1, "y": 1}).to_list(10000)
    df = pd.DataFrame(campaigns)
    
    # Sample data for performance
    if len(df) > 500:
        df = df.sample(500)
    
    return df.to_dict('records')

@api_router.get("/filters/jobs")
async def get_job_list(current_user: str = Depends(get_current_user)):
    jobs = await db.campaigns.distinct('job')
    return sorted(jobs)

@api_router.get("/filters/months")
async def get_month_list(current_user: str = Depends(get_current_user)):
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months = await db.campaigns.distinct('month')
    return sorted(months, key=lambda x: month_order.index(x) if x in month_order else 12)

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_subscription(data: PredictionRequest, current_user: str = Depends(get_current_user)):
    if ml_model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")
    
    try:
        # Prepare features
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
        
        feature_dict = data.model_dump()
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in label_encoders:
                try:
                    feature_dict[feature] = label_encoders[feature].transform([feature_dict[feature]])[0]
                except:
                    feature_dict[feature] = 0
        
        # Create feature array
        X = np.array([feature_dict[f] for f in categorical_features + numerical_features]).reshape(1, -1)
        
        # Scale numerical features
        X[0, len(categorical_features):] = scaler.transform(X[0, len(categorical_features):].reshape(1, -1))
        
        # Predict
        prediction = ml_model.predict(X)[0]
        probability = ml_model.predict_proba(X)[0][1]
        
        return PredictionResponse(
            prediction="yes" if prediction == 1 else "no",
            probability=round(float(probability), 3)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    await load_bank_data()
    await train_ml_model()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
