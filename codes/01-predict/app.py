# app.py
import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import gaussian_kde
import os

# ================== 配置 ==================
MODEL_DIR = r"F:/codes/00-WORK/work-1/01-myself/01-predict/01-model"
PLOT_DIR = r"F:/codes/00-WORK/work-1/01-myself/01-predict/01-plot"

REGRESSION_NAMES = ["AD", "TC", "S_TC", "CS", "S_CS"]
CLASSIFICATION_NAMES = ["HD_LD", "I", "II", "III", "IV"]

# ================== 页面样式 ==================
st.set_page_config(page_title="RPUF prediction platform", page_icon=":bar_chart:")
st.markdown("""
<style>
.main-container {padding: 20px; background-color: #f8f9fa;}
.sidebar {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
.feature-section {margin-bottom: 20px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;}
.slider-label {color: #4a5568; margin-top: 5px;}
.model-result {background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 15px;}
</style>
""", unsafe_allow_html=True)

# ================== 辅助：安全 key 名称 ==================
def safe_key(name: str) -> str:
    # 把任意字符串转换为只含字母数字和下划线的安全 key
    s = re.sub(r'[^0-9a-zA-Z]+', '_', name)
    return s

# ================== 编码函数 ==================
def build_dipoce(di_choice: str, po_types_list: List[str], ce_choice: str) -> str:
    DI_MAP = {'MDI': '01', 'PM-200': '10'}
    CE_MAP = {'none': '00', 'diol': '10', 'triol': '01'}
    PO_ORDER = [
        "Petroleum based polyether",
        "Petroleum based polyester",
        "Biobased polyether",
        "Biobased polyester",
        "Wood biomass",
        "Food industry waste"
    ]
    di_code = DI_MAP.get(di_choice, '00')
    po_set = set(po_types_list or [])
    po_bits = ['1' if p in po_set else '0' for p in PO_ORDER]
    po_code = ''.join(po_bits)
    ce_code = CE_MAP.get(ce_choice, '00')
    dipoce_str = di_code + po_code + ce_code
    return dipoce_str

def build_ba_bits_from_types(ba_types: List[str]) -> str:
    types = list(ba_types or [])
    while len(types) < 2:
        types.append('none')
    bits = ['0','0','0','0']
    for idx, t in enumerate(types[:2]):
        t = (t or 'none').lower()
        pos = 2 * idx
        if t == 'chemical':
            bits[pos] = '1'
        elif t == 'physical':
            bits[pos + 1] = '1'
    return ''.join(bits)

def build_cata_bits_from_types(cata_types: List[str]) -> str:
    types = list(cata_types or [])
    while len(types) < 3:
        types.append('none')
    bits = ['0'] * 6
    for idx, t in enumerate(types[:3]):
        t = (t or 'none').lower()
        pos = 2 * idx
        if t == 'amine':
            bits[pos] = '1'
        elif t == 'organometallic':
            bits[pos + 1] = '1'
    return ''.join(bits)

def build_ba_bits_from_names(ba_list: List[str]) -> str:
    types = []
    for name in (ba_list or []):
        nl = (name or '').lower()
        if 'chem' in nl or '化学' in name:
            types.append('chemical')
        elif 'phys' in nl or '物理' in name:
            types.append('physical')
        else:
            types.append('none')
    while len(types) < 2:
        types.append('none')
    return build_ba_bits_from_types(types)

def build_cata_bits_from_names(cata_list: List[str]) -> str:
    types = []
    for name in (cata_list or []):
        nl = (name or '').lower()
        if 'amine' in nl or '胺' in name:
            types.append('amine')
        elif 'metal' in nl or 'organometallic' in nl or '金属' in name:
            types.append('organometallic')
        else:
            types.append('none')
    while len(types) < 3:
        types.append('none')
    return build_cata_bits_from_types(types)

# ================== OneHot编码映射 ==================
TC_METHOD_MAP = {"GHP": 0, "HFM": 1, "TLS": 2, "TPS": 3}
CS_TD_MAP = {"parallel": 0, "vertical": 1}

# ================== 模型加载 ==================
@st.cache_resource
def load_models():
    regression_models = {}
    classification_models = {}
    for name in REGRESSION_NAMES:
        try:
            regression_models[name] = joblib.load(f"{MODEL_DIR}/{name}.joblib")
        except Exception as e:
            st.error(f"无法加载回归模型 '{name}.joblib'：{e}")
            return None, None
    for name in CLASSIFICATION_NAMES:
        try:
            classification_models[name] = joblib.load(f"{MODEL_DIR}/{name}.joblib")
        except Exception as e:
            st.error(f"无法加载分类模型 '{name}.joblib'：{e}")
            return None, None
    return regression_models, classification_models

# ================== 特征构造（回归/分类） ==================
def process_input_features_regression(inputs: dict, model_type: str) -> pd.DataFrame:
    dipoce = build_dipoce(inputs['DI'], inputs.get('PO_types', []), inputs['CE'])
    ba_code = build_ba_bits_from_types(inputs.get('BA_types', [])) if 'BA_types' in inputs else build_ba_bits_from_names(inputs.get('BA', []))
    cata_code = build_cata_bits_from_types(inputs.get('Cata_types', [])) if 'Cata_types' in inputs else build_cata_bits_from_names(inputs.get('Cata', []))
    base = {
        "DIPOCE": dipoce,
        "BA_Code": ba_code,
        "Cata_Code": cata_code,
        "DI_NCO": inputs["DI_NCO"],
        "PO_HV": inputs["PO_HV"],
        "PO_f": inputs["PO_f"],
        "BA_Mn": inputs["BA_Mn"],
        "PO_Phr": inputs["PO_Phr"],
        "BA_Phr": inputs["BA_Phr"],
        "FS_Phr": inputs["FS_Phr"],
        "Cata_Phr": inputs["Cata_Phr"],
        "f(H2O)": inputs["f(H2O)"],
        "M_loss": inputs["M_loss"],
        "Yield": inputs["Yield"],
        "R": inputs["R"],
        "CHS": inputs["CHS"],
        "A_Mix_t": inputs["A_Mix_t"],
        "AB_Mix_t": inputs["AB_Mix_t"],
        "Q": inputs["Q"],
        "MCS": inputs["MCS"],
        "Closed_CC": inputs["Closed_CC"]
    }
    if model_type in ["TC", "S_TC"]:
        base.update({"TC_T": inputs["TC_T"], "TC_Method": TC_METHOD_MAP.get(inputs["TC_Method"], 0)})
    if model_type in ["CS", "S_CS"]:
        base.update({"CS_rate": inputs["CS_rate"], "CS_TD": CS_TD_MAP.get(inputs["CS_TD"], 0)})
    return pd.DataFrame([base])

def process_input_features_classification(inputs: dict) -> pd.DataFrame:
    dipoce = build_dipoce(inputs['DI'], inputs.get('PO_types', []), inputs['CE'])
    ba_code = build_ba_bits_from_types(inputs.get('BA_types', [])) if 'BA_types' in inputs else build_ba_bits_from_names(inputs.get('BA', []))
    cata_code = build_cata_bits_from_types(inputs.get('Cata_types', [])) if 'Cata_types' in inputs else build_cata_bits_from_names(inputs.get('Cata', []))
    feats = {
        "DIPOCE": dipoce,
        "DI_NCO": inputs["DI_NCO"],
        "PO_HV": inputs["PO_HV"],
        "PO_f": inputs["PO_f"],
        "BA_Code": ba_code,
        "BA_Mn": inputs["BA_Mn"],
        "PO_Phr": inputs["PO_Phr"],
        "BA_Phr": inputs["BA_Phr"],
        "FS_Phr": inputs["FS_Phr"],
        "Cata_Phr": inputs["Cata_Phr"],  
        "f(H2O)": inputs["f(H2O)"],
        "M_loss": inputs["M_loss"],
        "Yield": inputs["Yield"],
        "R": inputs["R"],
        "CHS": inputs["CHS"],
        "Q": inputs["Q"],
        "Cata_Code": cata_code,
        "MCS": inputs["MCS"],
        "Closed_CC": inputs["Closed_CC"]
    }
    return pd.DataFrame([feats])

# ================== 分布可视化（带 KDE） ==================
def plot_all_distributions(results_dict):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), dpi=600)  # 提高分辨率
    axes = axes.flatten()

    for i, (feature_name, value) in enumerate(results_dict.items()):
        ax = axes[i]
        dist_file = os.path.join(PLOT_DIR, f"{feature_name}.npy")

        if os.path.exists(dist_file):
            values = np.load(dist_file)
        else:
            values = np.random.normal(0.5, 0.15, 500)

        kde = gaussian_kde(values)
        x_grid = np.linspace(min(values), max(values), 500)
        y_grid = kde(x_grid)

        ax.plot(x_grid, y_grid, lw=2, color="skyblue")
        ax.fill_between(x_grid, y_grid, color="skyblue", alpha=0.4)

        if value >= min(values) and value <= max(values):
            y_star = kde(value)
        else:
            y_star = 0
        ax.plot(value, y_star, marker="*", markersize=12, color="red",
                label=f"Pred {value:.3f}")

        ax.set_xlabel(feature_name, fontsize=10)
        # 每行第一个子图显示ylabel
        if i % 3 == 0:
            ax.set_ylabel("Density", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    axes[-1].axis("off")
    fig.tight_layout()
    return fig

# ================== 主函数 ==================
def main():
    st.set_page_config(layout="wide")
    st.title("🧪RPUF prediction platform for Apparent Density • Thermal Conductivity • Compressive Strength 📈")
    # st.markdown("---")

    # 只在这些特征显示单位；其他特征显示无单位（只写名字）
    feature_units = {
        "DI_NCO": "%",
        "PO_HV": "mg KOH/g",
        "BA_Mn": "g/mol",
        "f(H2O)": "mol",
        "Yield": "%",
        "CHS": "%",
        "A_Mix_t":"s",
        "AB_Mix_t": "s",
        "Q": "K∙h",
        "MCS": "μm",
        "Closed_CC": "%",
        "CS_rate":"mm/s",
        "TC_T":"K",
    }

    regression_models, classification_models = load_models()
    if regression_models is None or classification_models is None:
        st.warning("模型加载失败或路径不正确，请检查 MODEL_DIR 与模型文件。")
        return

    # ============ 特征输入区 ============
    with st.expander("Feature input area", expanded=True):
        # st.subheader("Category characteristics")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            di = st.selectbox("DI", ["MDI", "PM-200"], key="sel_di")
        with c2:
            po_types = st.multiselect("PO", [
                "Petroleum based polyether",
                "Petroleum based polyester",
                "Biobased polyether",
                "Biobased polyester",
                "Wood biomass",
                "Food industry waste"
            ], key="ms_po_types")
        with c3:
            ce = st.selectbox("CE", ["none", "diol", "triol"], key="sel_ce")
        with c4:
            ba1_type = st.selectbox("BA1", ["none", "chemical", "physical"], key="sel_ba1")
        with c5:
            ba2_type = st.selectbox("BA2", ["none", "chemical", "physical"], key="sel_ba2")

        c1, c2, c3, c4, c5 = st.columns(5)
        cata1_type = c1.selectbox("Cata1", ["none", "amine", "organometallic"], key="sel_cata1")
        cata2_type = c2.selectbox("Cata2", ["none", "amine", "organometallic"], key="sel_cata2")
        cata3_type = c3.selectbox("Cata3", ["none", "amine", "organometallic"], key="sel_cata3")
        CS_TD = c4.selectbox("CS_TD", ["parallel", "vertical"], key="sel_cs_td")
        TC_Method = c5.selectbox("TC_Method", ["GHP", "HFM", "TLS", "TPS"], key="sel_tc_method")

        # 数值型分组（Composition / Structural / Process）
        numeric_groups = {
            "Composition": {
                "DI_NCO": (30.0, 33.6, 31.5),
                "PO_HV": (200, 1701, 476),
                "PO_f": (2.0, 16.0, 4.9),
                "BA_Mn": (37.1, 148.1, 37.7),
                "PO_Phr": (0.65, 465.0, 100.0),
                "BA_Phr": (0.0, 30.5, 6.1),
                "FS_Phr": (0.011, 4.3, 1.5),
                "Cata_Phr": (0.0, 6.5, 1.2),
                "f(H2O)": (0.0, 0.28, 0.12),
                "R": (0.7, 2.4, 1.2),
                # "CHS": (32.8, 76.8, 59.7)
            },
            "Structural": {
                "M_loss": (1.0, 68.8, 9.7),
                "Yield": (1.0, 100.0, 94.1),
                "MCS": (20.0, 900.0, 301.8),
                "Closed_CC": (9.0, 99.0, 74.8),
            },
            "Process": {
                "Q": (-48.0, 1368.0, 135.62),
                "A_Mix_t": (10.0, 1200.0, 164.7),
                "AB_Mix_t": (5.0, 35.0, 13.6),
                "CS_rate": (0.00164, 0.006667, 0.002204),
                "TC_T": (283.15, 343.15, 296.5)
            }
        }

        # 用于保存数值输入
        values = {}
        for group_name, features in numeric_groups.items():
            # st.subheader(group_name)
            keys = list(features.keys())
            for i in range(0, len(keys), 5):
                cols = st.columns(5)
                for j, key in enumerate(keys[i:i+5]):
                    low, high, default = features[key]
                    label = key
                    unit = feature_units.get(key, None)
                    if unit:
                        label = f"{key} ({unit})"
                    # 安全 key
                    widget_key = "num_" + safe_key(key)
                    with cols[j]:
                        # number_input 保证 default 在范围内
                        values[key] = st.number_input(label, value=float(default), min_value=float(low), max_value=float(high), key=widget_key)

    # ============ 点击预测 ============
    if st.button("Prediction"):
        inputs = {
            "DI": di,
            "PO_types": po_types,
            "CE": ce,
            "BA_types": [ba1_type, ba2_type],
            "Cata_types": [cata1_type, cata2_type, cata3_type],
            "CS_TD": CS_TD,
            "TC_Method": TC_Method,
            **values
        }
        inputs["CHS"] = 100 / (100 + values["PO_Phr"])


        # ==== 回归预测 ====
        regression_results = {}
        for metric, model in regression_models.items():
            X_reg = process_input_features_regression(inputs, metric)
            try:
                pred = model.predict(X_reg)[0]
                regression_results[metric] = float(pred)
            except Exception as e:
                regression_results[metric] = f"Prediction failed: {e}"
                st.error(f"regression model {metric} Prediction anomaly: {e}")

        # ==== 分类预测 ====
        classification_results = {}
        X_cls = process_input_features_classification(inputs)
        for name, model in classification_models.items():
            try:
                proba = model.predict_proba(X_cls)[0]
                positive_prob = proba[1] if len(proba) >= 2 else proba[0]
                classification_results[name] = {"prob": float(positive_prob), "label": ("1" if positive_prob > 0.5 else "0")}
            except Exception as e:
                classification_results[name] = {"prob": None, "label": "分类失败"}
                st.error(f"classification model {name} Prediction anomaly: {e}")

        # 预测总览
        st.subheader("🏆 Results")
        all_models = ["I", "II", "III", "IV", "HD_LD", "AD", "TC", "S_TC", "CS", "S_CS"]
        result_row = []

        for m in all_models:
            if m in classification_results:
                result_row.append(classification_results[m]["label"])
            elif m in regression_results:
                result_row.append(f"{regression_results[m]:.4f}" if isinstance(regression_results[m], float) else "-")
            else:
                result_row.append("-")

        df_display = pd.DataFrame([["Result"] + result_row], columns=["Label"] + all_models)
        st.dataframe(df_display.style.hide(axis="index"), use_container_width=True)

        # 回归指标可视化
        st.subheader("📊 Plot")
        valid_results = {k: v for k, v in regression_results.items() if isinstance(v, float)}
        if valid_results:
            fig = plot_all_distributions(valid_results)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
