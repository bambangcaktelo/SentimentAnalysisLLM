import os
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
from fastapi_mail import ConnectionConfig, FastMail, MessageSchema
from dotenv import load_dotenv

load_dotenv()
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# --- Local Imports ---
from .tasks import run_full_analysis_pipeline, generate_chat_response, generate_reasoning_report
from .database import users_collection, results_store_collection
from .auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    create_special_token,
    verify_special_token,
    EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES,
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES
)

DAILY_LIMIT = 5

app = FastAPI(
    title="Sentiment Analysis API",
    description="An API for sentiment analysis with user accounts and history management.",
    version="4.6.0" # Version bump for new login feature
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Email Configuration ---
conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS=os.getenv("MAIL_STARTTLS", "True").lower() == "true",
    MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS", "False").lower() == "true",
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

fm = FastMail(conf)

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class TokenRequest(BaseModel):
    identifier: str # Can be username or email
    password: str
    
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AnalysisRequest(BaseModel):
    query: str

class TaskResponse(BaseModel):
    job_id: str

class ResultResponse(BaseModel):
    job_id: str
    status: str
    result: Dict[str, Any] | None = None
    questions_remaining: int

class ChatResponse(BaseModel):
    response: str
    questions_remaining: int

# --- Background Task Wrapper ---
async def run_analysis_background(query: str, job_id: str):
    print(f"Background task {job_id} started for query: '{query}'")
    try:
        result_data = await run_full_analysis_pipeline(query)
        await results_store_collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "completed", "result": result_data}}
        )
        print(f"Background task {job_id} completed successfully.")
    except Exception as e:
        print(f"Background task {job_id} failed: {e}")
        await results_store_collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "failed", "result": {"error": str(e)}}}
        )

# --- Quota Management Dependency ---
async def check_and_update_limit(current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    user = await users_collection.find_one({"username": username})
    today_utc = datetime.now(timezone.utc).date()

    last_prompt = user.get("last_prompt_date")
    last_prompt_date = last_prompt.date() if isinstance(last_prompt, datetime) else None

    if not last_prompt_date or last_prompt_date != today_utc:
        await users_collection.update_one({"username": username}, {"$set": {"prompt_count": 0}})
        user["prompt_count"] = 0

    if user.get("prompt_count", 0) >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"Daily limit of {DAILY_LIMIT} reasoning prompts reached.")

    new_count = user.get("prompt_count", 0) + 1
    await users_collection.update_one(
        {"username": username},
        {"$set": {"prompt_count": new_count, "last_prompt_date": datetime.now(timezone.utc)}}
    )
    return DAILY_LIMIT - new_count

# --- Authentication Endpoints ---
@app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Auth"])
async def register(user: UserCreate, background_tasks: BackgroundTasks):
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    if await users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_data = {
        "username": user.username,
        "email": user.email,
        "hashed_password": get_password_hash(user.password),
        "is_verified": False,
        "prompt_count": 0,
        "last_prompt_date": None
    }
    await users_collection.insert_one(user_data)
    
    token = create_special_token(user.username, "email_verification", EMAIL_VERIFICATION_TOKEN_EXPIRE_MINUTES)
    verification_link = f"{FRONTEND_URL}/verify-email?token={token}"
    message = MessageSchema(
        subject="Verify Your Account",
        recipients=[user.email],
        body=f"Please click the link to verify your account: {verification_link}",
        subtype="html"
    )
    background_tasks.add_task(fm.send_message, message)
    return {"message": "User created. Please check your email to verify your account."}

@app.post("/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: TokenRequest):
    # Check if the identifier is an email or a username
    if "@" in form_data.identifier:
        user = await users_collection.find_one({"email": form_data.identifier})
    else:
        user = await users_collection.find_one({"username": form_data.identifier})
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username/email or password")
    if not user.get("is_verified"):
        raise HTTPException(status_code=403, detail="Account not verified. Please check your email.")
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/verify-email", tags=["Auth"])
async def verify_email(token: str):
    username = verify_special_token(token, "email_verification")
    if not username:
        raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
    result = await users_collection.update_one({"username": username}, {"$set": {"is_verified": True}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found for verification.")
    return {"message": "Email verified successfully. You can now log in."}

@app.post("/forgot-password", tags=["Auth"])
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    user = await users_collection.find_one({"email": request.email})
    if user:
        token = create_special_token(user["username"], "password_reset", PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
        reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
        message = MessageSchema(
            subject="Password Reset Request", recipients=[request.email],
            body=f"Click the link to reset your password: {reset_link}", subtype="html"
        )
        background_tasks.add_task(fm.send_message, message)
    return {"message": "If an account with that email exists, a password reset link has been sent."}

@app.post("/reset-password", tags=["Auth"])
async def reset_password(request: ResetPasswordRequest):
    username = verify_special_token(request.token, "password_reset")
    if not username:
        raise HTTPException(status_code=400, detail="Invalid or expired password reset token.")
    hashed_password = get_password_hash(request.new_password)
    await users_collection.update_one({"username": username}, {"$set": {"hashed_password": hashed_password}})
    return {"message": "Password has been reset successfully."}

# --- History Endpoints ---
@app.get("/history", response_model=List[dict], tags=["History"])
async def get_history(current_user: dict = Depends(get_current_user)):
    history_cursor = results_store_collection.find(
        {"owner": current_user["username"], "status": "completed"},
        {"_id": 0, "job_id": 1, "result.query": 1, "created_at": 1}
    ).sort("created_at", -1).limit(50)
    return await history_cursor.to_list(length=50)

@app.delete("/history/single/{job_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["History"])
async def delete_history_item(job_id: str, current_user: dict = Depends(get_current_user)):
    result = await results_store_collection.delete_one({"job_id": job_id, "owner": current_user["username"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="History item not found.")
    return

@app.delete("/history/all", status_code=status.HTTP_204_NO_CONTENT, tags=["History"])
async def delete_all_history(current_user: dict = Depends(get_current_user)):
    await results_store_collection.delete_many({"owner": current_user["username"]})
    return

# --- Analysis Endpoints ---
@app.post("/guest-analyze", response_model=TaskResponse, status_code=202, tags=["Analysis"])
async def guest_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    initial_data = {
        "job_id": job_id, "owner": "guest", "status": "processing",
        "result": None, "created_at": datetime.now(timezone.utc)
    }
    await results_store_collection.insert_one(initial_data)
    background_tasks.add_task(run_analysis_background, request.query, job_id)
    return {"job_id": job_id}

@app.post("/analyze", response_model=TaskResponse, status_code=202, tags=["Analysis"])
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    job_id = str(uuid.uuid4())
    initial_data = {
        "job_id": job_id, "owner": current_user["username"], "status": "processing",
        "result": None, "created_at": datetime.now(timezone.utc)
    }
    await results_store_collection.insert_one(initial_data)
    background_tasks.add_task(run_analysis_background, request.query, job_id)
    return {"job_id": job_id}

@app.get("/guest-status/{job_id}", response_model=ResultResponse, tags=["Analysis"])
async def get_guest_task_status(job_id: str):
    result = await results_store_collection.find_one({"job_id": job_id, "owner": "guest"})
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found or it belongs to a registered user.")
    return {"job_id": job_id, "status": result["status"], "result": result.get("result"), "questions_remaining": DAILY_LIMIT}

@app.get("/status/{job_id}", response_model=ResultResponse, tags=["Analysis"])
async def get_task_status(job_id: str, current_user: dict = Depends(get_current_user)):
    result = await results_store_collection.find_one({"job_id": job_id, "owner": current_user["username"]})
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found or permission denied.")
    
    user = await users_collection.find_one({"username": current_user["username"]})
    today_utc = datetime.now(timezone.utc).date()
    last_prompt = user.get("last_prompt_date")
    last_prompt_date = last_prompt.date() if isinstance(last_prompt, datetime) else None
    current_prompt_count = user.get("prompt_count", 0) if last_prompt_date and last_prompt_date == today_utc else 0
    questions_remaining = DAILY_LIMIT - current_prompt_count
    
    return {"job_id": job_id, "status": result["status"], "result": result.get("result"), "questions_remaining": questions_remaining}

@app.post("/generate-reasoning/{job_id}", status_code=200, tags=["Analysis"])
async def start_reasoning_generation(job_id: str, current_user: dict = Depends(get_current_user)):
    analysis_data = await results_store_collection.find_one({"job_id": job_id, "owner": current_user["username"]})
    if not analysis_data or analysis_data["status"] != "completed":
        raise HTTPException(status_code=404, detail="Completed analysis not found.")
    if analysis_data["result"].get("reasoning_report"):
        return {"message": "Reasoning report already exists."}
    reasoning_report = await generate_reasoning_report(analysis_data["result"]["query"], analysis_data["result"]["rag2_results"])
    await results_store_collection.update_one({"job_id": job_id}, {"$set": {"result.reasoning_report": reasoning_report}})
    return {"message": "Reasoning report generated successfully."}

@app.post("/chat/{job_id}", response_model=ChatResponse, tags=["Chat"])
async def handle_chat(job_id: str, chat_request: ChatRequest, current_user: dict = Depends(get_current_user), questions_remaining: int = Depends(check_and_update_limit)):
    analysis_data = await results_store_collection.find_one({"job_id": job_id, "owner": current_user["username"]})
    if not analysis_data or not analysis_data["result"].get("reasoning_report"):
        raise HTTPException(status_code=400, detail="Reasoning report must be generated before starting a chat.")
    chat_history_for_llm = [msg.dict() for msg in chat_request.messages]
    ai_response = await generate_chat_response(analysis_data["result"], chat_history_for_llm)
    new_chat_history = chat_history_for_llm + [{"role": "assistant", "content": ai_response}]
    await results_store_collection.update_one({"job_id": job_id}, {"$set": {"result.chat_history": new_chat_history}})
    return ChatResponse(response=ai_response, questions_remaining=questions_remaining)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

