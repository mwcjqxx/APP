import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app_core as core

st.set_page_config(
    page_title='Osteoporosis Risk Prediction Model',
    page_icon='\U0001FA7A',
    layout='wide',
    initial_sidebar_state='collapsed',
)

CSS = """
<style>
[data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stFooter"], [data-testid="stBottom"], [data-testid="stMainMenu"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {
    display: none !important;
}
#MainMenu {display: none !important;}
footer {display: none !important;}
.block-container {padding-top: .7rem; padding-bottom: .7rem; max-width: 1400px;}
[data-testid="stAppViewContainer"] {background: #f4f7fa;}

[data-testid="stVerticalBlock"] {gap: 0 !important;}
.element-container {margin-bottom: .72rem !important;}
[data-testid="stHorizontalBlock"] {gap: 1.1rem !important;}

[data-testid="column"] {
    box-sizing: border-box;
    background: #ffffff; border: 1px solid #e2e9f1; border-radius: 12px;
    padding: 1.1rem 1.2rem 1.2rem 1.2rem;
    box-shadow: 0 2px 10px rgba(31, 78, 121, .07);
}

.header-title {
    text-align: center; font-family: "Times New Roman", Georgia, serif;
    font-size: 1.55rem; font-weight: 700; color: #16222e;
    letter-spacing: .01em; margin-bottom: .1rem; line-height: 1.2;
}
.header-sub {
    text-align: center; font-family: "Times New Roman", Georgia, serif;
    font-size: .98rem; color: #4d6479; margin-bottom: .05rem;
}
.header-meta {
    text-align: center; font-size: .84rem; color: #8494a3;
    letter-spacing: .015em; margin-bottom: .2rem;
}
.header-rule {
    border: none; border-top: 2px solid #1f4e79; width: 360px;
    margin: .1rem auto .45rem auto;
}

.section-title {
    font-size: 1rem; letter-spacing: .08em; text-transform: uppercase;
    color: #1f4e79; font-weight: 700; border-bottom: 2px solid #1f4e79;
    padding-bottom: .3rem; margin-bottom: .85rem;
}
.risk-card {border-radius: 10px; padding: .6rem .95rem; margin-top: .1rem;
            border: 1px solid transparent;}
.risk-low {border-left: 6px solid #2a9d8f; background: #f4fbf9; border-color: #d6ede7;}
.risk-high {border-left: 6px solid #c0392b; background: #fdf6f5; border-color: #f1d8d4;}
.risk-label {font-size: 1.05rem; color: #3b4a58; line-height: 1.35;}
.risk-sub {font-size: .84rem; color: #6d7b89; margin-top: .05rem;}
.prob-caption {font-size: .95rem; color: #3b4a58; margin-top: .25rem;}
.prob-num {font-size: 2.25rem; font-weight: 700; font-family: "Times New Roman", serif;
           line-height: 1.1;}
.risk-low .prob-num {color: #1f7a6f;}
.risk-high .prob-num {color: #b03a2e;}
.summary {font-size: .88rem; color: #6d7b89; margin-top: .3rem; line-height: 1.45;}
.note {font-size: .84rem; color: #55636f; margin-top: .3rem; line-height: 1.5;}
.note-meta {font-size: .78rem; color: #8a97a3; margin-top: .15rem; letter-spacing: .01em;}
.stale {
    font-size: .88rem; color: #8a5a1a; background: #FDF7EA;
    border-left: 4px solid #D9A93B; border-radius: 6px;
    padding: .35rem .6rem; margin: .2rem 0 .05rem 0;
}
.footer-left {
    font-size: .86rem; color: #8594a2; text-align: left; border-top: 1px solid #e8ecf1;
    padding-top: .4rem; margin-top: .5rem; line-height: 1.5;
}

.stNumberInput, .stRadio {margin-bottom: 0 !important;}
.stNumberInput label, .stRadio > label {
    font-size: .96rem !important; font-weight: 500; color: #2c3a47;
    margin-bottom: .15rem !important; padding-bottom: 0 !important;
}
.stNumberInput div[data-baseweb="input"] {
    max-height: 2.3rem; min-height: 2.3rem; border-radius: 7px;
    border-color: #c9d6e2; background: #ffffff;
}
.stNumberInput div[data-baseweb="input"]:focus-within {
    border-color: #1f4e79; box-shadow: 0 0 0 1px #1f4e79;
}
.stNumberInput div[data-baseweb="input"] input {font-size: 1rem; font-weight: 500;}

.stRadio div[role="radiogroup"] {gap: .55rem !important; padding-top: .05rem;}
.stRadio div[role="radiogroup"] label {
    font-size: .95rem !important; font-weight: 400 !important;
    margin-bottom: 0 !important; color: #2c3a47;
}
.stButton {margin-top: .45rem; margin-bottom: .5rem;}
.stButton button {
    width: 100%; font-size: 1.05rem; height: 2.5rem; font-weight: 600;
    border-radius: 7px; border: none; color: #ffffff; background: #1f4e79;
    box-shadow: 0 2px 8px rgba(31, 78, 121, .25); padding: 0;
}
.stButton button:hover {background: #17415f; color: #ffffff;
                        box-shadow: 0 4px 14px rgba(31, 78, 121, .35);}
.stButton button:focus:not(:active) {color: #ffffff;}
.stInfo {
    background: #f0f6fb; border: 1px solid #d5e3f0; border-left: 4px solid #1f4e79;
    border-radius: 7px; padding: .6rem .8rem; font-size: .92rem; color: #33475b;
}
.stInfo p {margin: 0;}

[data-testid="stImage"] {max-width: 90%; margin-left: auto; margin-right: auto;}
[data-testid="stImage"] img {max-width: 100%; height: auto;}
[data-testid="stTooltipIcon"] svg {fill: #8ba0b5;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def get_bundle():
    core.setup_style()
    model = core.load_model()
    art = core.load_assets()
    explainer = core.make_explainer(model, core.load_background())
    model.predict_proba(core.to_frame({f['name']: float(f['default'])
                                       for f in art['features']}))
    return model, art, explainer

model, art, explainer = get_bundle()
F = {f['name']: f for f in art['features']}
THR = float(art['threshold'])
perf = art['performance']

def _num(x, fb):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(fb)
    return v if np.isfinite(v) else float(fb)

st.markdown(
    "<div class='header-title'>Osteoporosis Risk Prediction in Type 2 Diabetes: "
    "An Interpretable Machine Learning Model</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='header-meta'>XGBoost &middot; %d predictors &middot; AUC %.3f "
    "(internal) / %.3f (external) &middot; Youden threshold %.3f &middot; "
    "For research use only</div>"
    % (art['feature_count'], perf['internal']['auc'], perf['external']['auc'], THR),
    unsafe_allow_html=True)
st.markdown("<hr class='header-rule'>", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1.15], gap='medium')

with left_col:
    st.markdown("<div class='section-title'>Variables</div>", unsafe_allow_html=True)

    gender = st.radio(
        'Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female',
        index=int(F['Gender']['default']), horizontal=True)
    age = st.number_input('Age (years)', value=int(F['Age']['default']), step=1)
    muscle = st.number_input('Muscle mass, DXA (g)',
                             value=int(F['Muscle_Mass']['default']), step=100)
    rsmi = st.number_input('RSMI (kg/m\u00b2)', value=float(F['RSMI']['default']),
                           step=0.01, format='%.2f')
    hba1c = st.number_input('HbA1c (%)', value=float(F['HbA1c']['default']),
                            step=0.01, format='%.2f')
    bap = st.number_input('Bone-specific ALP (\u00b5g/L)',
                          value=float(F['BAP']['default']), step=0.01, format='%.2f')
    pinp = st.number_input('Total PINP (ng/mL)', value=float(F['Total_PINP']['default']),
                           step=0.01, format='%.2f')
    ag = st.number_input('Android-to-gynoid fat ratio', value=float(F['A/G']['default']),
                         step=0.01, format='%.2f')

values = {
    'Gender': float(gender),
    'Age': _num(age, F['Age']['default']),
    'Muscle_Mass': _num(muscle, F['Muscle_Mass']['default']),
    'RSMI': _num(rsmi, F['RSMI']['default']),
    'HbA1c': _num(hba1c, F['HbA1c']['default']),
    'BAP': _num(bap, F['BAP']['default']),
    'Total_PINP': _num(pinp, F['Total_PINP']['default']),
    'A/G': _num(ag, F['A/G']['default']),
}
signature = tuple(round(values[c], 4) for c in core.SELECTED)

with right_col:
    st.markdown("<div class='section-title'>Prediction Result</div>",
                unsafe_allow_html=True)

    predict_clicked = st.button('Predict')

    if predict_clicked:
        with st.spinner('Computing prediction and SHAP explanation ...'):
            t0 = time.time()
            prob = core.predict_proba(model, values)
            t_pred = time.time() - t0
            t1 = time.time()
            sv, base, fx = core.shap_local(explainer, values)
            t_shap = time.time() - t1
        st.session_state['result'] = {
            'signature': signature, 'values': dict(values), 'prob': prob,
            'sv': sv, 'base': base, 'fx': fx, 't_pred': t_pred, 't_shap': t_shap,
        }

    res = st.session_state.get('result')

    if res is None:
        st.info('Enter patient features on the left, then click **Predict**.')
    else:
        prob = res['prob']
        pos = prob >= THR
        st.markdown(
            "<div class='risk-card %s'>"
            "<div class='risk-label'>Predicted outcome: <b>Osteoporosis-%s</b></div>"
            "<div class='risk-sub'>%s the Youden threshold %.3f</div>"
            "<div class='prob-caption'>Probability of osteoporosis:</div>"
            "<div class='prob-num'>%.1f%%</div>"
            "</div>"
            % ('risk-high' if pos else 'risk-low',
               'positive' if pos else 'negative',
               'above' if pos else 'below', THR, prob * 100),
            unsafe_allow_html=True)

        if res['signature'] != signature:
            st.markdown("<div class='stale'>Inputs changed &mdash; press <b>Predict</b> "
                        "to refresh.</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>SHAP Force Plot</div>",
                    unsafe_allow_html=True)
        fig = core.fig_force_plot(res['sv'], res['base'], res['fx'], res['values'],
                                  threshold=THR)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            "<div class='note'><b>SHAP values are on the probability scale, consistent with "
            "Figure 6 of the manuscript</b> (f(x) = E[f(X)] + \u03a3\u03c6). Screening result "
            "only; the predicted probability is not a calibrated individual risk.</div>"
            "<div class='note-meta'>Prediction %.0f ms &middot; SHAP explanation %.2f s"
            "</div>" % (res['t_pred'] * 1000, res['t_shap']),
            unsafe_allow_html=True)
