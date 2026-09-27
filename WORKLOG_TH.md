# บันทึกการพัฒนา UNSW-NB15 Web App

เอกสารนี้บันทึกสิ่งที่ทำ เหตุผล ปัญหาที่พบ การแก้ไข ผลตรวจสอบ และข้อจำกัด
ของ Web App แยกจาก Notebook ฝึกโมเดล

## 1. เป้าหมาย

สร้าง Web App หน้าเดียวที่นำเสนอผลงาน Machine Learning ได้ง่าย ผู้ใช้กรอก
ข้อมูล network flow แล้วรับผล `Normal` หรือ `Attack` โดย App ต้องใช้โมเดล
เดียวกับผลการทดลองและต้องไม่ฝึกโมเดลใหม่

## 2. ขอบเขตที่ตัดสินใจใช้

- Framework: Streamlit เพราะทำงานกับ Python และ scikit-learn โดยตรง
- Deployment target: Streamlit Community Cloud
- Model: `best_model.joblib` จาก Notebook การทดลอง
- Task: Binary Classification (`0 = Normal`, `1 = Attack`)
- UI: หน้าเดียว มีข้อมูลโมเดล ฟอร์ม input ผลการทำนาย และคำอธิบายข้อจำกัด
- ไม่รวม Dataset ขนาดใหญ่ใน Web App เพราะ inference ต้องใช้เพียง pipeline
  ที่ฝึกแล้ว

## 3. ไฟล์ที่สร้าง

### `app.py`

ส่วนติดต่อผู้ใช้ของ Streamlit ทำหน้าที่ดังนี้

1. โหลดโมเดลผ่านฟังก์ชัน cache เพียงครั้งเดียว
2. อ่าน feature schema จาก model bundle
3. สร้าง input widget ตามชนิด numeric/categorical โดยอัตโนมัติ
4. เรียง feature ให้ตรงกับลำดับตอนฝึก
5. เรียก `predict()` และ `predict_proba()` เท่านั้น
6. แสดงผล Normal/Attack พร้อม Attack probability
7. แสดง metrics จริงและคำเตือน Sample mode

### `model_runtime.py`

แยก logic ที่ไม่เกี่ยวกับ UI ออกจาก Streamlit เพื่อให้ทดสอบง่าย ประกอบด้วย

- การตรวจว่าไฟล์โมเดลมีอยู่จริง
- การตรวจ required keys ของ model bundle
- การตรวจลำดับ `feature_schema` กับ `selected_features`
- การสร้าง DataFrame หนึ่งแถวตามลำดับ feature ที่ถูกต้อง
- การหาตำแหน่งคลาส `1` จาก `pipeline.classes_` ก่อนอ่าน probability

### `requirements.txt`

กำหนด dependency สำหรับ deploy โดยล็อก `scikit-learn==1.6.1` เพราะค่า
`_sklearn_version` ภายในโมเดลระบุเวอร์ชันนี้โดยตรง ส่วน Streamlit, Pandas และ
Joblib จำกัด major version เพื่อลดความเสี่ยงจาก breaking change

### `.streamlit/config.toml`

กำหนดธีม สี และ headless server ให้หน้าตาสม่ำเสมอทั้ง local และ cloud

### `tests/`

- `test_model_runtime.py`: ตรวจ model contract, ลำดับคอลัมน์ และ prediction
- `test_app_smoke.py`: เปิด App ผ่าน Streamlit AppTest และกดปุ่ม Predict

## 4. โมเดลที่นำมาใช้

- Dataset: UNSW-NB15
- Split: official train/test split
- Run mode: Full
- Train หลังทำความสะอาด: 96,822 แถว
- Test หลังทำความสะอาด: 49,971 แถว
- Best model: Decision Tree
- Accuracy: 0.8058
- Precision (Attack): 0.6672
- Recall (Attack): 0.9405
- F1 (Attack): 0.7806

โมเดลถูกเลือกด้วย Attack F1 สูงสุด และใช้ Recall เป็นตัวตัดสินเมื่อ F1 เท่ากัน
ตามหลักที่กำหนดไว้ใน Notebook

## 5. เหตุผลที่สร้าง App ใหม่แทนการรันใน Notebook

- Colab runtime หยุดได้และ URL ภายในไม่เหมาะกับการส่งงาน
- Notebook ควรรับผิดชอบการทดลอง ส่วน Web App ควรรับผิดชอบ inference
- การแยกส่วนช่วยลดความเสี่ยงที่ App จะฝึกโมเดลซ้ำหรือได้ผลไม่ตรงรายงาน
- Repository ขนาดเล็ก deploy และตรวจสอบง่ายกว่า Notebook พร้อม Dataset
- อาจารย์เปิดลิงก์และทดลองได้โดยไม่ต้องรันเซลล์ตามลำดับ

## 6. ปัญหาที่ป้องกันไว้

### เวอร์ชัน scikit-learn ไม่ตรงกัน

ไฟล์ pickle/joblib อาจโหลดผิดพลาดหรือมีพฤติกรรมต่างกันเมื่อเวอร์ชันไม่ตรงกับ
ตอนฝึก จึงตรวจ metadata ในโมเดลและล็อก `scikit-learn==1.6.1`

### ลำดับ feature ผิด

โมเดลต้องรับคอลัมน์ในรูปแบบเดียวกับตอนฝึก จึงสร้าง DataFrame ด้วย
`selected_features` จาก bundle โดยตรง ไม่พึ่งลำดับ widget บนหน้าจอ

### Probability ผูกกับคอลัมน์ผิด

ไม่สมมติว่าคอลัมน์ที่สองของ `predict_proba()` เป็น Attack เสมอ แต่ค้นหา index
ของคลาส `1` จาก `pipeline.classes_`

### หมวดหมู่ใหม่

categorical preprocessing ที่ถูกบันทึกใน pipeline ใช้
`handle_unknown="ignore"` จึงไม่หยุดทำงานเมื่อพบค่าที่ไม่เคยเห็นระหว่างฝึก

### ไฟล์โมเดลหายหรือ bundle ไม่ครบ

App ตรวจ path และ required keys ก่อน render ฟอร์ม พร้อมแสดงข้อความที่ผู้ใช้
แก้ไขได้ แทนการปล่อย traceback ที่อ่านยาก

## 7. จุดอ่อนและข้อจำกัด

1. โมเดลปัจจุบันใช้ Full mode แต่ยังประเมินบนชุดข้อมูลทดลอง ไม่ใช่ traffic ขององค์กรจริง
2. ข้อมูล UNSW-NB15 มาจากสภาพแวดล้อมปี 2015 และอาจเกิด dataset drift
3. Decision Tree มีความเสี่ยง overfitting และ probability อาจไม่ calibrated
4. Binary classification ไม่บอกประเภทของการโจมตี
5. การกรอก feature ด้วยมือไม่เท่ากับระบบตรวจจับ intrusion แบบ real-time
6. ค่าเริ่มต้นของฟอร์มมาจากสถิติชุดฝึก ไม่ใช่ traffic ของผู้ใช้
7. ผล Normal ไม่ใช่การรับประกันว่าระบบปลอดภัย
8. ก่อนใช้งานจริงควรเพิ่ม authentication, logging, monitoring และ rate limit

## 8. สิ่งที่ต้องทำก่อนส่งงานฉบับสุดท้าย

1. รัน Notebook V2 แบบ Full mode
2. ดาวน์โหลด `best_model.joblib` ชุดใหม่
3. แทนที่ไฟล์ใน `artifacts/`
4. รันทดสอบทั้งหมดใหม่
5. อัปเดตตัวเลขในรายงานจาก `run_summary.json`
6. Deploy และทดสอบ URL จากอุปกรณ์อื่น
7. เก็บภาพหน้าจอหน้า App และผล Prediction สำหรับรายงาน

## 9. หลักการแปลผล

- Recall สูงช่วยลด Attack ที่โมเดลพลาด แต่ทำให้ False Positive เพิ่มได้
- Precision ประมาณ 0.6664 หมายความว่าคำเตือน Attack ยังมีส่วนที่เป็น
  False Positive จึงต้องตรวจสอบร่วมกับ log และบริบท
- ควรอธิบาย Accuracy, Precision, Recall, F1 และ Confusion Matrix ร่วมกัน
  ไม่ใช้ Accuracy เพียงค่าเดียว

## 10. การเปลี่ยนโมเดลในอนาคต

หาก Notebook ส่งออก model bundle ที่รักษา contract เดิม ได้แก่ `pipeline`,
`feature_schema`, `selected_features` และ `class_mapping` สามารถแทนที่
`artifacts/best_model.joblib` แล้วทดสอบใหม่ได้โดยไม่ต้องแก้ UI หลัก

## 11. ผลการตรวจสอบจริงก่อนส่งมอบ

วันที่ตรวจสอบ: 18 กันยายน 2026

สภาพแวดล้อมทดสอบถูกสร้างแยกในโฟลเดอร์ชั่วคราว และติดตั้ง dependency จาก
`requirements.txt` เพื่อไม่ให้ไฟล์ไลบรารีปะปนกับแพ็กเกจส่งมอบ ผลตรวจสอบมีดังนี้

1. ตรวจ syntax ของ `app.py`, `model_runtime.py` และ test files ผ่านทั้งหมด
2. โหลด `best_model.joblib` ด้วย `scikit-learn 1.6.1` สำเร็จ
3. ตรวจ model contract และลำดับ feature ผ่าน
4. สร้าง input หนึ่งแถวและทำนายด้วย pipeline จริงสำเร็จ
5. เปิด App ด้วย Streamlit AppTest แล้วกดปุ่มทำนายสำเร็จ
6. Unit/Smoke tests ผ่าน 4 จาก 4 รายการ ใช้เวลาประมาณ 12.24 วินาที
7. เปิด Streamlit server แบบ headless แล้ว health endpoint ตอบ HTTP 200 และ `ok`
8. ค้นหาใน `app.py` และ `model_runtime.py` แล้วไม่พบคำสั่ง `.fit(`
9. SHA-256 ของโมเดลใน Web App ตรงกับโมเดลต้นฉบับ:
   `5E27827CC7BD319CB1AFF58A562252023A7BECC055F58E1E1C329B99514ABA15`

การทดสอบค่าเริ่มต้นของฟอร์มให้ผล `Attack` และ Attack probability ประมาณ
0.9839 การตรวจนี้มีไว้ยืนยันว่าเส้นทาง inference ทำงานครบ ไม่ใช่ metric ใหม่
และไม่ควรนำไปใช้ประเมินประสิทธิภาพของโมเดล

## 12. ผลกระทบต่อผลการทดลอง

การสร้าง Web App ครั้งนี้ **ไม่เปลี่ยนผลการทดลองเดิม** เพราะไม่ได้เรียก `fit()`,
ไม่ได้เปลี่ยน preprocessing, hyperparameters, train/test split หรือข้อมูลฝึก
ไฟล์โมเดลเป็นสำเนาแบบ byte-for-byte จาก Notebook และผ่านการตรวจ SHA-256

สิ่งที่เพิ่มเป็นเพียงชั้นรับ input, ตรวจความครบถ้วน, เรียงคอลัมน์ และเรียก
`predict()`/`predict_proba()` จาก pipeline เดิม ดังนั้น Accuracy, Precision,
Recall และ F1 ที่แสดงยังเป็นค่าจากการทดลองเดิม ไม่ใช่ค่าที่ App คำนวณใหม่

## 13. สถานะพร้อมใช้งาน

- พร้อมรันบนเครื่องด้วย `streamlit run app.py`
- พร้อมอัปโหลดโฟลเดอร์นี้ขึ้น GitHub
- พร้อมเลือก `app.py` เป็น entrypoint บน Streamlit Community Cloud
- ยังไม่ได้เผยแพร่เป็น URL สาธารณะ เพราะขั้นตอนนั้นต้องเลือกบัญชี GitHub และ
  Streamlit ของผู้ใช้
- รายงานฉบับสมบูรณ์ใช้ผล Full mode ชุดเดียวกับ artifact ในโฟลเดอร์นี้
