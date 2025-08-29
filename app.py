# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
# ───────────────────────────────────────────────
# 1. สร้างแอป FastAPI
# ───────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0",
    description="API สำหรับทำนายความเสี่ยงสินเชื่อด้วย Random Forest"
)

# ───────────────────────────────────────────────
# 2. เปิดใช้งาน CORS
# ───────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────
# 3. โหลดโมเดล Random Forest
# ───────────────────────────────────────────────
try:
    model = joblib.load("random_forest_credit.pkl")
except Exception as e:
    raise RuntimeError(f"ไม่สามารถโหลดโมเดลได้: {e}")

# ───────────────────────────────────────────────
# 4. Schema สำหรับข้อมูลอินพุต
# ───────────────────────────────────────────────
class InputData(BaseModel):
    EXT_SOURCE_3: float = Field(..., ge=0, le=1, description="External source score 3 (0.0 - 1.0)")
    EXT_SOURCE_2: float = Field(..., ge=0, le=1, description="External source score 2 (0.0 - 1.0)")
    FLAG_PHONE: int = Field(..., ge=0, le=1, description="ลูกค้ามีเบอร์โทรศัพท์หรือไม่ (0 หรือ 1)")
    REG_CITY_NOT_WORK_CITY: int = Field(..., ge=0, le=1, description="ทำงานในเมืองที่ไม่ใช่ที่ลงทะเบียน (0 หรือ 1)")
    REGION_RATING_CLIENT: int = Field(..., ge=1, le=3, description="เรตติ้งภูมิภาคของลูกค้า (1 = ดี, 3 = แย่)")
    AMT_REQ_CREDIT_BUREAU_YEAR: float = Field(..., ge=0, description="จำนวนครั้งที่ขอเครดิตบูโรใน 1 ปีที่ผ่านมา")

# ───────────────────────────────────────────────
# 5. Health Check
# ───────────────────────────────────────────────
@app.get("/health", summary="API health check")
def health_check():
    return {"status": "ok", "model_loaded": True}

# ───────────────────────────────────────────────
# 6. ฟังก์ชันทำนาย
# ───────────────────────────────────────────────
@app.post("/predict", summary="ทำนายความเสี่ยงสินเชื่อ")
def predict(data: InputData):
    try:
        # จัดรูปข้อมูลให้ตรงกับฟีเจอร์ของโมเดล
        input_df = pd.DataFrame([[
            data.EXT_SOURCE_3,
            data.EXT_SOURCE_2,
            data.FLAG_PHONE,
            data.REG_CITY_NOT_WORK_CITY,
            data.REGION_RATING_CLIENT,
            data.AMT_REQ_CREDIT_BUREAU_YEAR
        ]], columns=[
            "EXT_SOURCE_3",
            "EXT_SOURCE_2",
            "FLAG_PHONE",
            "REG_CITY_NOT_WORK_CITY",
            "REGION_RATING_CLIENT",
            "AMT_REQ_CREDIT_BUREAU_YEAR"
        ])

        # ทำนาย
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),  # 0 = ปลอดภัย, 1 = เสี่ยง
            "probability_of_default": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
