import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="REE 多模型科研系统", layout="wide")
st.title("REE 多模型预测系统（完整科研终极版）")

# =====================================================
# 1️⃣ 基本信息
# =====================================================

st.sidebar.header("基本信息")

sex_label = st.sidebar.selectbox("性别", ["女", "男"])
sex = 0 if sex_label == "女" else 1

age = st.sidebar.number_input("年龄", value=25.0)
height = st.sidebar.number_input("身高(cm)", value=170.0)
weight = st.sidebar.number_input("体重(kg)", value=65.0)

# =====================================================
# 🏃 体力活动水平
# =====================================================

st.sidebar.markdown("### 🏃 体力活动水平")

pal_label = st.sidebar.selectbox(
    "请选择体力活动水平",
    [
        "低（基本不运动，久坐人群，每周≤1次）",
        "中（每周<3次，强度较低）",
        "高（每周≥3次中高强度运动）"
    ]
)

pal_map = {
    "低（基本不运动，久坐人群，每周≤1次）": 3,
    "中（每周<3次，强度较低）": 2,
    "高（每周≥3次中高强度运动）": 1
}
pal = pal_map[pal_label]

st.sidebar.info("""
低：几乎不运动  
中：每周少于3次轻度运动  
高：每周3次以上中高强度运动
""")

# =====================================================
# 2️⃣ 体成分输入
# =====================================================

st.sidebar.header("体成分指标")

FFM = st.sidebar.number_input("FFM", value=50.0)
BM = st.sidebar.number_input("BM", value=65.0)
BMC = st.sidebar.number_input("BMC", value=3.5)
LBM = st.sidebar.number_input("LBM", value=45.0)
ALST = st.sidebar.number_input("ALST", value=20.0)
FM = st.sidebar.number_input("FM", value=20.0)
BMI = st.sidebar.number_input("BMI", value=22.0)
fat = st.sidebar.number_input("体脂%", value=20.0)

height_m = height / 100

# =====================================================
# 3️⃣ 衍生指标（传统回归专用）
# =====================================================

AT = 1.18 * FM
BT = 1.85 * BMC * 1.0436
SMT = (1.13 * ALST) - (0.02 * age) + (0.61 * sex) + 0.97

brain_g = 920 + height * 2.7 if sex == 1 else 748 + height * 3.1
Brain = brain_g / 1000

Heart = (0.012 * LBM) ** 1.0499
Kidney = (0.0165 * LBM) ** 0.9306
Liver = (0.0778 * LBM) ** 0.9277
Spleen = (0.022 * LBM) ** 1.4449
Skin = 0.00949 * (BM ** 0.441) * (height_m ** 0.655) * 2 * 1.05

RTH = BM - (AT + BT + SMT + Brain)
RTW = Heart + Kidney + Liver + Spleen + Skin
RTwomen = RTH - RTW
RTmen = BM - (AT + BT + SMT + Brain + LBM + Skin)

# =====================================================
# ① 分层线性方程（有截距）
# =====================================================

if sex == 0:
    if pal == 1:
        REE_raw = 618.097 + 8.714*FFM + 13.280*LBM + 7.953*ALST
        raw_formula = "618.097 + 8.714*FFM + 13.280*LBM + 7.953*ALST"
    elif pal == 2:
        REE_raw = -315.846 + 20.149*ALST + 6.951*height + 3.940*FFM
        raw_formula = "-315.846 + 20.149*ALST + 6.951*height + 3.940*FFM"
    else:
        REE_raw = 587.619 + 19.221*ALST + 6.385*FFM + 6.731*LBM
        raw_formula = "587.619 + 19.221*ALST + 6.385*FFM + 6.731*LBM"
else:
    if pal == 1:
        REE_raw = 743.902 + 16.223*ALST + 5.962*FFM + 5.085*LBM
        raw_formula = "743.902 + 16.223*ALST + 5.962*FFM + 5.085*LBM"
    elif pal == 2:
        REE_raw = 575.477 + 17.043*LBM + 11.627*ALST + 80.531*BMC
        raw_formula = "575.477 + 17.043*LBM + 11.627*ALST + 80.531*BMC"
    else:
        REE_raw = 582.089 + 32.177*ALST + 34.719*BMC + 3.643*LBM
        raw_formula = "582.089 + 32.177*ALST + 34.719*BMC + 3.643*LBM"


# =====================================================
# ③ 器官组织简化方程
# =====================================================

if sex == 0:

    if pal == 1:

        EE_RTW = 61.302 * RTW
        EE_Liver = 122.843 * Liver
        EE_Kidney = 28.320 * Kidney

        REE_der_top3 = (
            946.221
            + 32.931 * EE_RTW
            - 23.707 * EE_Liver
            - 28.480 * EE_Kidney
        )

        der_top3_formula = (
            "946.221 + 32.931*(61.302*RTW)"
            " - 23.707*(122.843*Liver)"
            " - 28.480*(28.320*Kidney)"
        )

    elif pal == 2:

        EE_SMT = 19.695 * SMT
        EE_Brain = 433.599 * Brain
        EE_RTW = 47.863 * RTW

        REE_der_top3 = (
            -1774.508
            + 0.961 * EE_SMT
            + 4.706 * EE_Brain
            + 1.553 * EE_RTW
        )

        der_top3_formula = (
            "-1774.508 + 0.961*(19.695*SMT)"
            " + 4.706*(433.599*Brain)"
            " + 1.553*(47.863*RTW)"
        )

    else:

        EE_SMT = 14.196 * SMT
        EE_Liver = 118.570 * Liver
        EE_Kidney = 27.215 * Kidney

        REE_der_top3 = (
            582.218
            + 1.280 * EE_SMT
            + 3.114 * EE_Liver
            - 15.703 * EE_Kidney
        )

        der_top3_formula = (
            "582.218 + 1.280*(14.196*SMT)"
            " + 3.114*(118.570*Liver)"
            " - 15.703*(27.215*Kidney)"
        )

else:

    if pal == 1:

        EE_SMT = 17.504 * SMT
        EE_Skin = 20.098 * Skin
        EE_BT = 37.897 * BT

        REE_der_top3 = (
            636.907
            + 0.879 * EE_SMT
            + 67.208 * EE_Skin
            + 0.915 * EE_BT
        )

        der_top3_formula = (
            "636.907 + 0.879*(17.504*SMT)"
            " + 67.208*(20.098*Skin)"
            " + 0.915*(37.897*BT)"
        )

    elif pal == 2:

        EE_SMT = 13.720 * SMT
        EE_BT = 77.590 * BT
        EE_Skin = 148.963 * Skin

        REE_der_top3 = (
            219.361
            + 0.628 * EE_SMT
            + 0.862 * EE_BT
            + 22.187 * EE_Skin
        )

        der_top3_formula = (
            "219.361 + 0.628*(13.720*SMT)"
            " + 0.862*(77.590*BT)"
            " + 22.187*(148.963*Skin)"
        )

    else:

        EE_SMT = 28.795 * SMT
        EE_BT = 20.478 * BT
        EE_Skin = 47.892 * Skin

        REE_der_top3 = (
            477.957
            + 0.956 * EE_SMT
            + 1.028 * EE_BT
            + 17.827 * EE_Skin
        )

        der_top3_formula = (
            "477.957 + 0.956*(28.795*SMT)"
            " + 1.028*(20.478*BT)"
            " + 17.827*(47.892*Skin)"
        )

# =====================================================
# ④ 通用线性方程
# =====================================================

if sex == 0:
    REE_step = 94.405 + 6.224*FFM + 8.076*ALST + 44.083*BMC + 6.533*LBM - 20.808*pal +3.193*height+ 4.419*BMI
    step_formula = "611.829 + 6.910*FFM + 7.856*ALST + 48.407*BMC + 4.846*weight + 1.526*BM - 19.198*pal - 8.928*BMI"
else:
    REE_step = 297.811 + 8.536*ALST + 4.058*FFM + 9.284*LBM + 42.842*BMC - 19.660*pal + 1.379*BM + 2.257*height
    step_formula = "297.811 + 8.536*ALST + 4.058*FFM + 9.284*LBM + 42.842*BMC - 19.660*pal + 1.379*BM + 2.257*height"


# =====================================================
# ⑥ 机器学习
# =====================================================

EE_AT = 4.5 * AT
EE_Liver = 200 * Liver
EE_RTH = 43 * RTH

try:
    model = joblib.load("model.pkl")
    input_df = pd.DataFrame([{
        "性别": sex,
        "体力活动水平": pal,
        "年龄": age,
        "身高": height,
        "FFM": FFM,
        "BMC": BMC,
        "ALST": ALST,
        "BMI": BMI,
        "体脂%": fat,
        "EE_AT": EE_AT,
        "EE_Liver": EE_Liver,
        "EE_RTH": EE_RTH
    }])
    REE_ml = model.predict(input_df)[0]
except:
    REE_ml = np.nan

# =====================================================
# 模型展示
# =====================================================

df_models = pd.DataFrame({
    "模型": ["分层线性方程（有截距）", "器官组织简化方程", "通用线性方程", "机器学习"],
    "REE(kcal/d)": [REE_raw, REE_der_top3, REE_step, REE_ml],
    "公式": [raw_formula, der_top3_formula, step_formula, "ExtraTreesRegressor"]
})

st.header("📊 模型预测对比")
st.dataframe(df_models, use_container_width=True)



# =====================================================
# ⑦ 附加能量呈现指标（基于 Ridge 结果）
# =====================================================

st.header("🔥 附加能量指标")

# PAL 系数（注意这里是正确映射）
pal_coef_map = {
    "低（基本不运动，久坐人群，每周≤1次）": 1.3,
    "中（每周<3次，强度较低）": 1.5,
    "高（每周≥3次中高强度运动）": 1.7
}

pal_coef = pal_coef_map[pal_label]

# 推荐使用 Ridge 作为基准模型
REE_base = REE_ml

# 1️⃣ 每日总能量消耗
TEE = REE_base * pal_coef

# 2️⃣ 最低安全摄入
min_intake = REE_base * 0.8

# 3️⃣ 维持体重
maintain_intake = TEE

# 4️⃣ 健康减重推荐
weight_loss_intake = max(TEE - 500, min_intake)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("每日总能量消耗 (TEE)", f"{TEE:.0f} kcal")

with col2:
    st.metric("最低安全摄入", f"{min_intake:.0f} kcal")

with col3:
    st.metric("维持体重所需", f"{maintain_intake:.0f} kcal")

with col4:
    st.metric("健康减重推荐", f"{weight_loss_intake:.0f} kcal")


# =====================================================
# ⑧ 帮助 / 说明页
# =====================================================

st.header("📘 帮助与说明")

with st.expander("🔬 REE 是什么？"):
    st.write("""
静息能量消耗（Resting Energy Expenditure，REE）是人体在完全安静休息状态下，
维持呼吸、血液循环、细胞代谢等基本生命活动所需的最低能量。

在很多文献中也称为静息代谢率（Resting Metabolic Rate, RMR）。
从严格意义上讲，REE 与基础代谢率（BMR）并不完全相同，
RMR通常比BMR高约10%左右。

由于BMR测试条件较为严苛（需绝对空腹、恒温、清晨等条件），
实际研究和临床应用中通常以REE作为代替。
""")

with st.expander("📊 结果如何解读？"):
    st.write("""
一般情况下（排除病理因素）：

• 数值越高 → 代表代谢水平越旺盛  
• 数值越低 → 代表基础代谢偏低  

代谢水平受：
- 年龄
- 性别
- 肌肉量
- 内脏器官质量
- 体脂比例
等多因素影响。

本系统提供多模型结果对比，便于科研分析。
""")

with st.expander("⚠ 免责声明"):
    st.write("""
本工具仅用于科研分析与健康管理参考。

不用于疾病诊断或治疗决策，
不替代医生专业建议。

如有代谢异常、体重异常或相关疾病，
请咨询专业医疗人员。
""")