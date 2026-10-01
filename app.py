"""One-page Streamlit website for the fitted UNSW-NB15 model."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from model_runtime import (
    ModelBundleError,
    calculate_derived_features,
    load_model_bundle,
    predict_flow,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.joblib"

FEATURE_LABELS = {
    "dur": "Flow duration",
    "proto": "Protocol",
    "service": "Network service",
    "state": "Connection state",
    "spkts": "Source packets",
    "dpkts": "Destination packets",
    "sbytes": "Source bytes",
    "dbytes": "Destination bytes",
    "rate": "Packet rate",
    "sload": "Source load",
    "dload": "Destination load",
}

FEATURE_HELP = {
    "dur": "ระยะเวลารวมของ network flow",
    "proto": "โพรโทคอลที่ใช้ เช่น TCP หรือ UDP",
    "service": "บริการเครือข่าย เช่น HTTP, DNS หรือ FTP",
    "state": "สถานะการเชื่อมต่อของ flow",
    "spkts": "จำนวนแพ็กเก็ตจากต้นทางไปปลายทาง",
    "dpkts": "จำนวนแพ็กเก็ตจากปลายทางกลับต้นทาง",
    "sbytes": "จำนวนไบต์จากต้นทางไปปลายทาง",
    "dbytes": "จำนวนไบต์จากปลายทางกลับต้นทาง",
    "rate": "อัตราแพ็กเก็ตของ flow",
    "sload": "โหลดข้อมูลจากต้นทาง",
    "dload": "โหลดข้อมูลจากปลายทาง",
}

DERIVED_FEATURES = {"rate", "sload", "dload"}


st.set_page_config(
    page_title="UNSW-NB15 Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --ink:#0b1728; --muted:#607089; --line:#d8e2e9; --paper:#f5f7f8;
        --navy:#071421; --accent:#2457d6; --accent-soft:#e8eefc; --normal-soft:#e7f6ef;
        --danger:#9a2e40; --danger-soft:#fff0f2;
      }
      .stApp {
        background:linear-gradient(rgba(11,23,40,.035) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(11,23,40,.035) 1px,transparent 1px),var(--paper);
        background-size:32px 32px; color:var(--ink);
        font-family:"Leelawadee UI",Tahoma,sans-serif;
      }
      .block-container { max-width:1180px; padding:1.4rem 2rem 3.5rem; }
      header[data-testid="stHeader"] { background:transparent; }
      footer,[data-testid="stSidebar"] { display:none; }
      h1,h2,h3,p { letter-spacing:-.015em; }

      .topbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.65rem 0 1.1rem; }
      .brand { display:flex; align-items:center; gap:.7rem; font-size:.9rem; font-weight:800; letter-spacing:.08em; }
      .brand-mark { width:34px; height:34px; display:grid; place-items:center; border:1px solid var(--ink); font-size:.8rem; }
      .status-pill { display:inline-flex; align-items:center; gap:.45rem; padding:.38rem .65rem; border:1px solid #b9c7d2; background:#ffffffcc; font-size:.78rem; font-weight:700; }
      .status-dot { width:7px; height:7px; border-radius:50%; background:#15803d; box-shadow:0 0 0 4px rgba(21,128,61,.12); }

      .hero { position:relative; overflow:hidden; display:grid; grid-template-columns:1.3fr .7fr; gap:2.4rem; min-height:390px; padding:3.2rem; color:white; background:var(--navy); border:1px solid #20384a; }
      .eyebrow { color:#9db6ff; font-size:.78rem; font-weight:700; letter-spacing:.04em; }
      .hero h1 { max-width:720px; margin:.85rem 0 1rem; font-size:clamp(2.5rem,6vw,4.7rem); line-height:.98; letter-spacing:-.055em; }
      .hero-copy { max-width:650px; margin:0; color:#c5d2dd; font-size:1.05rem; line-height:1.75; }
      .hero-side { position:relative; z-index:1; align-self:end; border-left:1px solid #ffffff2e; padding-left:1.5rem; }
      .hero-side-label { margin-bottom:.5rem; color:#91a6b7; font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
      .hero-model { margin-bottom:1.3rem; font-size:1.4rem; font-weight:750; }
      .hero-route { display:grid; gap:.45rem; color:#d8e3ea; font-size:.85rem; }
      .hero-route span::before { content:"→"; margin-right:.55rem; color:#9db6ff; }

      .section-head { display:flex; align-items:end; justify-content:space-between; gap:1.5rem; margin:3.4rem 0 1rem; padding-bottom:.8rem; border-bottom:1px solid #bbc8d2; }
      .section-title { margin:.3rem 0 0; color:var(--ink); font-size:clamp(1.55rem,3vw,2.15rem); line-height:1.1; }
      .section-note { max-width:430px; margin:0; color:var(--muted); font-size:.88rem; line-height:1.55; text-align:right; }

      .metric-grid { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); background:white; }
      .metric-card { min-height:132px; padding:1.25rem 1.35rem; border-right:1px solid var(--line); }
      .metric-card:last-child { border-right:0; }
      .metric-label { color:var(--muted); font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
      .metric-value { margin-top:.55rem; color:var(--ink); font-family:Bahnschrift,"Segoe UI",sans-serif; font-size:2.15rem; font-weight:700; letter-spacing:-.025em; }
      .metric-foot { margin-top:.25rem; color:var(--muted); font-size:.75rem; }
      .sample-banner { margin-top:.8rem; padding:.8rem 1rem; border-left:3px solid #d49732; background:#fff8e9; color:#6c4a0c; font-size:.86rem; }

      div[data-testid="stVerticalBlockBorderWrapper"] { padding:.35rem; border-color:var(--line); border-radius:0; background:white; }
      div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"] p { color:#25374b; font-size:.84rem; font-weight:700; }
      .stNumberInput input,div[data-baseweb="select"]>div { border-radius:0!important; }
      .stButton button,.stFormSubmitButton button { min-height:3rem; border:1px solid var(--accent)!important; border-radius:0!important; background:var(--accent)!important; color:white!important; font-weight:800!important; }
      .stButton button:hover,.stFormSubmitButton button:hover { border-color:#1d46ad!important; background:#1d46ad!important; }
      input:focus-visible,button:focus-visible,[role="combobox"]:focus-visible { outline:3px solid rgba(36,87,214,.28)!important; outline-offset:2px!important; }

      .derived-shell { margin:.35rem 0 1rem; border:1px solid var(--line); background:#f7f9fb; }
      .derived-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.8rem 1rem; border-bottom:1px solid var(--line); }
      .derived-title { color:var(--ink); font-size:.82rem; font-weight:800; }
      .derived-note { color:var(--muted); font-size:.72rem; }
      .derived-grid { display:grid; grid-template-columns:repeat(3,1fr); }
      .derived-item { min-width:0; padding:.85rem 1rem; border-right:1px solid var(--line); }
      .derived-item:last-child { border-right:0; }
      .derived-label { color:var(--muted); font-size:.72rem; font-weight:700; }
      .derived-value { margin-top:.25rem; overflow-wrap:anywhere; color:var(--ink); font-family:Bahnschrift,"Segoe UI",sans-serif; font-size:1rem; font-weight:700; }
      .derived-unit { margin-top:.12rem; color:var(--muted); font-size:.66rem; }

      .result-shell { min-height:100%; padding:1.6rem; border:1px solid var(--line); background:white; }
      .result-index { color:var(--muted); font-size:.72rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
      .result-empty { padding:2.2rem 0; color:var(--muted); font-size:.95rem; line-height:1.65; }
      .result-state { margin:1rem 0 1.2rem; padding:1.4rem; border:1px solid; }
      .result-state.normal { border-color:#80cbbd; background:var(--normal-soft); color:#07594f; }
      .result-state.attack { border-color:#e3a0aa; background:var(--danger-soft); color:var(--danger); }
      .result-name { font-size:2.1rem; font-weight:850; letter-spacing:-.04em; }
      .result-desc { margin-top:.45rem; line-height:1.6; }
      .interpretation { margin-top:1rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted); font-size:.84rem; line-height:1.6; }

      .workflow { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); background:white; }
      .workflow-step { min-height:180px; padding:1.35rem; border-right:1px solid var(--line); }
      .workflow-step:last-child { border-right:0; }
      .workflow-number { color:var(--accent); font-size:.75rem; font-weight:850; letter-spacing:.12em; }
      .workflow-step h3 { margin:2.2rem 0 .55rem; font-size:1rem; }
      .workflow-step p { margin:0; color:var(--muted); font-size:.82rem; line-height:1.55; }
      .footer { display:grid; grid-template-columns:1fr auto; gap:1rem; margin-top:3.5rem; padding:1.5rem 0 .5rem; border-top:1px solid #bbc8d2; color:var(--muted); font-size:.78rem; line-height:1.5; }

      @media(max-width:820px){
        .block-container{padding:.75rem 1rem 2.5rem}.hero{grid-template-columns:1fr;min-height:auto;padding:2rem 1.4rem}.hero-side{border-left:0;border-top:1px solid #ffffff2e;padding:1.2rem 0 0}.metric-grid,.workflow{grid-template-columns:1fr 1fr}.metric-card:nth-child(2),.workflow-step:nth-child(2){border-right:0}.metric-card:nth-child(-n+2),.workflow-step:nth-child(-n+2){border-bottom:1px solid var(--line)}.section-head{align-items:start;flex-direction:column}.section-note{text-align:left}
        .derived-head{align-items:flex-start;flex-direction:column;gap:.2rem}.derived-grid{grid-template-columns:1fr}.derived-item{border-right:0;border-bottom:1px solid var(--line)}.derived-item:last-child{border-bottom:0}
      }
      @media(max-width:520px){
        .topbar{align-items:flex-start}.metric-grid,.workflow{grid-template-columns:1fr}.metric-card,.workflow-step{border-right:0;border-bottom:1px solid var(--line)}.metric-card:last-child,.workflow-step:last-child{border-bottom:0}.hero h1{font-size:2.55rem}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def get_model_bundle(model_path: str):
    return load_model_bundle(Path(model_path))


try:
    bundle = get_model_bundle(str(MODEL_PATH))
except ModelBundleError as exc:
    st.error(str(exc))
    st.stop()

model_name = escape(str(bundle.get("model_name", "Machine Learning model")))
metrics = bundle.get("metrics", {})
run_mode_raw = str(bundle.get("run_mode", "ไม่ระบุ"))
run_mode = escape(run_mode_raw)
schema = bundle["feature_schema"]

st.markdown(
    f"""
    <div class="topbar">
      <div class="brand"><span class="brand-mark">NB</span> UNSW-NB15 LAB</div>
      <div class="status-pill"><span class="status-dot"></span> Model loaded · {run_mode}</div>
    </div>
    <section class="hero">
      <div>
        <div class="eyebrow">Machine Learning for Network Security</div>
        <h1>ตรวจจับความเสี่ยงจาก Network Flow</h1>
        <p class="hero-copy">เว็บไซต์สาธิตการจำแนกข้อมูลเครือข่ายเป็น Normal หรือ Attack ด้วยโมเดลที่ฝึกจากชุดข้อมูล UNSW-NB15 พร้อมแสดงหลักฐานการประเมินและข้อจำกัดของผลลัพธ์</p>
      </div>
      <div class="hero-side">
        <div class="hero-side-label">Active model</div>
        <div class="hero-model">{model_name}</div>
        <div class="hero-side-label">Inference route</div>
        <div class="hero-route"><span>8 inputs + 3 calculated features</span><span>Fitted preprocessing pipeline</span><span>Binary prediction</span></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)


def metric_value(name: str) -> str:
    value = metrics.get(name)
    return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"


st.markdown(
    f"""
    <div class="section-head">
      <div><h2 class="section-title">ผลการประเมินบนชุดทดสอบ</h2></div>
      <p class="section-note">Attack ถูกกำหนดเป็น positive class จึงต้องอ่าน Precision, Recall และ F1 ร่วมกัน</p>
    </div>
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{metric_value('Accuracy')}</div><div class="metric-foot">ความถูกต้องโดยรวม</div></div>
      <div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">{metric_value('Precision')}</div><div class="metric-foot">ความแม่นยำของคำเตือน</div></div>
      <div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value">{metric_value('Recall')}</div><div class="metric-foot">สัดส่วน Attack ที่ตรวจพบ</div></div>
      <div class="metric-card"><div class="metric-label">F1 score</div><div class="metric-value">{metric_value('F1')}</div><div class="metric-foot">สมดุล Precision และ Recall</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if run_mode_raw == "sample":
    st.markdown(
        '<div class="sample-banner"><strong>Sample mode:</strong> ผลชุดนี้เหมาะสำหรับการพัฒนาและสาธิต ยังไม่ควรใช้แทนผลจาก Full dataset</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-head">
      <div><h2 class="section-title">วิเคราะห์ Network Flow</h2></div>
      <p class="section-note">กรอกค่าจาก network flow หนึ่งรายการ ระบบจะใช้ pipeline เดียวกับการทดลองเพื่อประเมินผล</p>
    </div>
    """,
    unsafe_allow_html=True,
)

form_column, result_column = st.columns([1.28, .72], gap="large")

with form_column:
    values: dict[str, object] = {}
    input_schema = [feature for feature in schema if feature["name"] not in DERIVED_FEATURES]
    with st.container(border=True):
        for row_start in range(0, len(input_schema), 2):
            row_columns = st.columns(2)
            for column_index, feature in enumerate(input_schema[row_start : row_start + 2]):
                name = feature["name"]
                label = FEATURE_LABELS.get(name, name)
                help_text = FEATURE_HELP.get(name)
                with row_columns[column_index]:
                    if feature["kind"] == "categorical":
                        options = feature.get("choices") or ["Unknown"]
                        default = feature.get("default", options[0])
                        default_index = options.index(default) if default in options else 0
                        values[name] = st.selectbox(label, options, index=default_index, help=help_text)
                    else:
                        default = max(0.0, float(feature.get("default", 0.0)))
                        step = max(float(feature.get("step", 1.0)), 0.000001)
                        values[name] = st.number_input(
                            label, min_value=0.0, value=default, step=step,
                            format="%.6f", help=help_text,
                        )

        values.update(calculate_derived_features(values))
        st.markdown(
            f"""
            <div class="derived-shell">
              <div class="derived-head">
                <span class="derived-title">ค่าที่ระบบคำนวณอัตโนมัติ</span>
                <span class="derived-note">คำนวณใหม่ทันทีเมื่อข้อมูลด้านบนเปลี่ยน</span>
              </div>
              <div class="derived-grid">
                <div class="derived-item">
                  <div class="derived-label">Packet rate</div>
                  <div class="derived-value">{values['rate']:,.6f}</div>
                  <div class="derived-unit">packets / second</div>
                </div>
                <div class="derived-item">
                  <div class="derived-label">Source load</div>
                  <div class="derived-value">{values['sload']:,.6f}</div>
                  <div class="derived-unit">bits / second</div>
                </div>
                <div class="derived-item">
                  <div class="derived-label">Destination load</div>
                  <div class="derived-value">{values['dload']:,.6f}</div>
                  <div class="derived-unit">bits / second</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        submitted = st.button(
            "วิเคราะห์ Network Flow",
            type="primary",
            use_container_width=True,
            key="analyze_flow",
        )

if submitted:
    try:
        prediction, attack_probability = predict_flow(bundle, values)
    except Exception as exc:
        st.session_state["prediction_error"] = str(exc)
        st.session_state.pop("prediction_result", None)
    else:
        st.session_state["prediction_error"] = None
        st.session_state["prediction_result"] = {
            "prediction": prediction,
            "attack_probability": attack_probability,
        }
        st.session_state["prediction_count"] = st.session_state.get("prediction_count", 0) + 1

with result_column:
    prediction_error = st.session_state.get("prediction_error")
    prediction_result = st.session_state.get("prediction_result")

    if prediction_error:
        st.error(f"ไม่สามารถทำนายได้: {prediction_error}")
    elif prediction_result is None:
        st.markdown(
            """
            <div class="result-shell">
              <div class="result-index">Prediction output</div>
              <div class="result-empty"><strong>รอข้อมูลสำหรับวิเคราะห์</strong><br><br>ตรวจค่าที่กรอกทางซ้าย แล้วกดปุ่ม “วิเคราะห์ Network Flow” ผลลัพธ์จะปรากฏในพื้นที่นี้</div>
              <div class="interpretation">Normal หมายถึงรูปแบบใกล้เคียง traffic ปกติ ส่วน Attack หมายถึงรูปแบบใกล้เคียงตัวอย่างการโจมตีในข้อมูลฝึก</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        prediction = prediction_result["prediction"]
        attack_probability = prediction_result["attack_probability"]
        prediction_count = st.session_state.get("prediction_count", 1)

        if prediction == 1:
            state_class, state_name = "attack", "Attack"
            state_desc = "รูปแบบ flow นี้ใกล้เคียงตัวอย่างการโจมตีในข้อมูลฝึก ควรตรวจสอบ log ต้นฉบับและบริบทของระบบเพิ่มเติม"
        else:
            state_class, state_name = "normal", "Normal"
            state_desc = "รูปแบบ flow นี้ใกล้เคียง traffic ปกติในข้อมูลฝึก แต่ผลนี้ไม่ใช่การรับประกันว่าระบบปลอดภัย"

        probability_text = f"{attack_probability:.2%}" if attack_probability is not None else "N/A"
        st.markdown(
            f"""
            <div class="result-shell">
              <div class="result-index">Prediction output · ครั้งที่ {prediction_count}</div>
              <div class="result-state {state_class}"><div class="result-name">{state_name}</div><div class="result-desc">{state_desc}</div></div>
              <div class="metric-label">Attack probability</div><div class="metric-value">{probability_text}</div>
              <div class="interpretation">โมเดลที่ใช้: {model_name}<br>เปลี่ยนค่าแล้วกดวิเคราะห์ซ้ำได้ทันที โดยไม่ต้องรีเฟรชหน้าเว็บ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if attack_probability is not None:
            st.progress(min(max(int(round(attack_probability * 100)), 0), 100))

st.markdown(
    """
    <div class="section-head">
      <div><h2 class="section-title">จากข้อมูลดิบสู่ผลทำนาย</h2></div>
      <p class="section-note">หน้าเว็บเรียกใช้โมเดลที่ฝึกแล้ว ไม่มีการฝึกโมเดลใหม่ระหว่างการทำนาย</p>
    </div>
    <div class="workflow">
      <div class="workflow-step"><div class="workflow-number">01</div><h3>Flow input</h3><p>รับข้อมูลพื้นฐาน 8 ค่า แล้วคำนวณ rate และ load อีก 3 ค่าโดยอัตโนมัติ</p></div>
      <div class="workflow-step"><div class="workflow-number">02</div><h3>Preprocessing</h3><p>เติมค่าที่ขาด ปรับสเกลตัวเลข และเข้ารหัสข้อมูลหมวดหมู่ด้วย pipeline เดิม</p></div>
      <div class="workflow-step"><div class="workflow-number">03</div><h3>Classification</h3><p>ส่งข้อมูลเข้าสู่โมเดลที่ผ่านการเปรียบเทียบบนชุดทดสอบแล้ว</p></div>
      <div class="workflow-step"><div class="workflow-number">04</div><h3>Interpretation</h3><p>แสดง Normal หรือ Attack พร้อมความน่าจะเป็นและคำอธิบายข้อจำกัด</p></div>
    </div>
    <div class="footer"><div><strong>UNSW-NB15 Intrusion Detection</strong><br>Educational machine learning prototype</div><div>Binary classification · 0 Normal / 1 Attack</div></div>
    """,
    unsafe_allow_html=True,
)
