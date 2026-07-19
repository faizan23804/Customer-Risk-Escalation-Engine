import streamlit as st # type: ignore
import requests
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore

#  Page config 
st.set_page_config(
    page_title = "Customer Risk Engine",
    page_icon  = "🚨",
    layout     = "wide"
)

API_URL = "http://localhost:8000"

#  Header 
st.title("🚨 Customer Risk & Escalation Engine")
st.markdown("**Multimodal ML system — Tabular + NLP Late Fusion**")
st.divider()

#  Health check 
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health['pipeline_loaded']:
        st.success("Model API is running")
    else:
        st.error("❌ Pipeline not loaded")
except Exception:
    st.error("❌ Cannot connect to API — make sure FastAPI is running on port 8000")
    st.stop()

st.divider()

#  Sidebar — Ticket Input Form 
st.sidebar.title("📋 New Support Ticket")
st.sidebar.markdown("Fill in ticket details to get escalation prediction")

with st.sidebar:
    customer_segment = st.selectbox(
        "Customer Segment",
        ["individual", "small_business", "enterprise",
         "education", "non_profit"]
    )
    product_area = st.selectbox(
        "Product Area",
        ["billing", "api_integration", "login_auth",
         "mobile_app", "analytics_dashboard",
         "data_export", "notifications"]
    )
    issue_type = st.selectbox(
        "Issue Type",
        ["account_access", "billing_problem", "bug",
         "feature_request", "how_to", "other",
         "performance", "security_concern"]
    )
    priority = st.selectbox(
        "Priority",
        ["low", "medium", "high", "urgent"]
    )
    status = st.selectbox(
        "Status",
        ["open", "in_progress", "on_hold",
         "resolved", "closed_no_action"]
    )
    sla_plan = st.selectbox(
        "SLA Plan",
        ["standard", "gold", "platinum"]
    )
    platform = st.selectbox(
        "Platform",
        ["web", "android", "ios", "desktop"]
    )
    region = st.selectbox(
        "Region",
        ["NA", "EU", "APAC", "MEA", "LATAM"]
    )
    customer_sentiment = st.selectbox(
        "Customer Sentiment",
        ["very_positive", "positive", "neutral",
         "negative", "very_negative"]
    )
    csat_score = st.slider(
        "CSAT Score", 
        min_value=0, max_value=5, value=3
    )
    resolution_time = st.number_input(
        "Resolution Time (hours) — leave 0 if unresolved",
        min_value=0.0, value=0.0
    )
    reopened = st.selectbox(
        "Was ticket reopened?", 
        [0, 1], 
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    has_attachment = st.selectbox(
        "Has Attachment?", 
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    st.markdown("---")
    st.markdown("**Ticket Text**")

    initial_message = st.text_area(
        "Customer Message",
        value="I cannot access my account and nobody is helping me.",
        height=100
    )
    agent_reply = st.text_area(
        "Agent First Reply",
        value="We are looking into this issue.",
        height=80
    )
    resolution_summary = st.text_area(
        "Resolution Summary (leave blank if unresolved)",
        value="",
        height=80
    )

    predict_btn = st.button(
        "🔍 Predict Escalation Risk",
        use_container_width=True,
        type="primary"
    )

#  Main Area — Results 
if predict_btn:
    payload = {
        "customer_segment"     : customer_segment,
        "product_area"         : product_area,
        "issue_type"           : issue_type,
        "priority"             : priority,
        "status"               : status,
        "sla_plan"             : sla_plan,
        "initial_message"      : initial_message,
        "agent_first_reply"    : agent_reply,
        "resolution_summary"   : resolution_summary if resolution_summary else None,
        "resolution_time_hours": resolution_time if resolution_time > 0 else None,
        "reopened"             : reopened,
        "customer_sentiment"   : customer_sentiment,
        "csat_score"           : csat_score,
        "has_attachment"       : has_attachment,
        "platform"             : platform,
        "region"               : region,
        "created_at"           : "2026-07-15T10:30:00"
    }

    with st.spinner("Analyzing ticket..."):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json    = payload,
                timeout = 60
            )
            result = response.json()

            #  Risk Score Display 
            col1, col2, col3 = st.columns(3)

            risk_level = result['risk_level']
            risk_score = result['risk_score']
            escalated  = result['escalated']

            # Color mapping
            color_map = {
                "Critical": "🔴",
                "High"    : "🟠",
                "Medium"  : "🟡",
                "Low"     : "🟢"
            }
            bg_color_map = {
                "Critical": "#FF4B4B",
                "High"    : "#FF8C00",
                "Medium"  : "#FFD700",
                "Low"     : "#00CC44"
            }

            with col1:
                st.metric(
                    label = "Risk Score",
                    value = f"{risk_score:.1%}"
                )

            with col2:
                st.markdown(
                    f"""
                    <div style='background-color:{bg_color_map[risk_level]};
                                padding:20px; border-radius:10px;
                                text-align:center;'>
                        <h2 style='color:white; margin:0;'>
                            {color_map[risk_level]} {risk_level}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col3:
                st.metric(
                    label = "Escalation Recommended",
                    value = "YES ⚠️" if escalated else "NO ✅"
                )

            st.divider()

            #  Risk Gauge 
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = risk_score * 100,
                title = {"text": "Escalation Risk Score (%)"},
                gauge = {
                    "axis" : {"range": [0, 100]},
                    "bar"  : {"color": bg_color_map[risk_level]},
                    "steps": [
                        {"range": [0,  40], "color": "#E8F5E9"},
                        {"range": [40, 60], "color": "#FFF9C4"},
                        {"range": [60, 80], "color": "#FFE0B2"},
                        {"range": [80, 100],"color": "#FFEBEE"}
                    ],
                    "threshold": {
                        "line" : {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.divider()

            #  SHAP Explanation 
            st.subheader("🔍 Why This Risk Score?")

            reasons = result['top_reasons']

            if reasons:
                features = [r['feature'] for r in reasons]
                impacts  = [float(r['impact']) for r in reasons]
                colors   = [
                    "#FF4B4B" if v > 0 else "#00CC44"
                    for v in impacts
                ]

                fig_shap = go.Figure(go.Bar(
                    x           = impacts,
                    y           = features,
                    orientation = 'h',
                    marker_color= colors
                ))
                fig_shap.update_layout(
                    title  = "Feature Impact on Risk Score",
                    xaxis_title = "SHAP Value (impact on prediction)",
                    yaxis_title = "Feature",
                    height = 400
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                #  Reasons table 
                st.subheader("📋 Top Risk Factors")
                reasons_df = pd.DataFrame(reasons)
                st.dataframe(
                    reasons_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            #  Model Info 
            st.subheader("🤖 Model Information")
            info_col1, info_col2 = st.columns(2)

            with info_col1:
                st.info(f"**Model Used:** {result['model_used']}")

            with info_col2:
                try:
                    model_info = requests.get(
                        f"{API_URL}/model-info"
                    ).json()
                    st.info(
                        f"**Features:** {model_info['total_features']} "
                        f"({model_info['tabular_features']} tabular + "
                        f"{model_info['embedding_dim']} NLP embeddings)"
                    )
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

else:
    #  Default state 
    st.info("👈 Fill in the ticket details in the sidebar and click **Predict Escalation Risk**")

    st.markdown("""
    ### How This System Works
    
    This multimodal ML system combines two types of analysis:
    
    **📊 Tabular Analysis**
    - Customer segment, product area, priority
    - Resolution time, SLA plan, ticket status
    - Historical patterns from 100K support tickets
    
    **📝 NLP Analysis (DistilBERT)**  
    - Customer message sentiment and intent
    - Agent response patterns
    - Resolution summary content
    
    **🔀 Late Fusion**
    - Both signals combined into 780 features
    - LightGBM makes final escalation prediction
    - SHAP explains every individual prediction
    """)