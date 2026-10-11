import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent/"src"))
from modeling import clean_data, find_target, evaluate_models, permutation_importance_table

st.set_page_config(page_title="E-Commerce Churn Intelligence", page_icon="🛒", layout="wide")
st.title("🛒 E-Commerce Customer Churn Prediction System")
st.caption("End-to-end churn analytics, ML comparison, explainability and retention risk scoring.")

uploaded = st.file_uploader("Upload a customer CSV (or use the included demo dataset)", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else pd.read_csv("data/customer_churn_demo.csv")
df = clean_data(df)
target = find_target(df)

st.sidebar.header("Controls")
if target is None:
    st.error("No churn target found. Expected a column such as Churn, Exited or Attrition.")
    st.stop()
threshold = st.sidebar.slider("Churn decision threshold", 0.10, 0.90, 0.50, 0.01)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Customers", f"{len(df):,}")
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
rate = pd.to_numeric(df[target], errors="coerce").mean() if df[target].dtype!="object" else None
c4.metric("Churn rate", f"{rate:.1%}" if rate is not None else "Categorical")

tab1,tab2,tab3,tab4 = st.tabs(["EDA","Model Lab","Explainability","Risk Scoring"])
with tab1:
    st.subheader("Data quality")
    st.dataframe(pd.DataFrame({"dtype":df.dtypes.astype(str),"missing":df.isna().sum(),"unique":df.nunique()}),
                 use_container_width=True)
    left,right=st.columns(2)
    with left:
        if target in df.columns:
            counts = df[target].astype(str).value_counts().reset_index()
            counts.columns=["Class","Count"]
            st.plotly_chart(px.bar(counts,x="Class",y="Count",title="Target distribution"),use_container_width=True)
    with right:
        nums=df.select_dtypes(include=np.number)
        if not nums.empty:
            st.plotly_chart(px.imshow(nums.corr(numeric_only=True),text_auto=".2f",title="Numeric correlation"),
                            use_container_width=True)
    st.dataframe(df.head(25), use_container_width=True)

with tab2:
    st.subheader("Compare classification models")
    results, fitted = evaluate_models(df,target,threshold)
    st.dataframe(results.style.format({c:"{:.3f}" for c in results.columns if c!="Model"}), use_container_width=True)
    st.plotly_chart(px.bar(results.melt(id_vars="Model",value_vars=["Precision","Recall","F1","ROC-AUC","PR-AUC"]),
                           x="Model",y="value",color="variable",barmode="group",title="Model performance"),
                    use_container_width=True)
    best = results.iloc[0]["Model"]
    st.success(f"Recommended model by F1/ROC-AUC: **{best}**")

with tab3:
    st.subheader("Most important churn factors")
    results, fitted = evaluate_models(df,target,threshold)
    best = results.iloc[0]["Model"]
    pipe, Xt, yt, proba, pred = fitted[best]
    imp = permutation_importance_table(pipe,Xt,yt)
    st.plotly_chart(px.bar(imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h",
                           title=f"Permutation importance — {best}"),use_container_width=True)
    st.info("High importance means changing that feature most affected validation F1. It is an association, not proof of causation.")

with tab4:
    st.subheader("Customer-level churn risk")
    results, fitted = evaluate_models(df,target,threshold)
    best = results.iloc[0]["Model"]
    pipe = fitted[best][0]
    Xall = df.drop(columns=[target])
    ids = Xall["Customer_ID"] if "Customer_ID" in Xall.columns else pd.Series(np.arange(len(Xall)),name="Customer_ID")
    Xin = Xall.drop(columns=["Customer_ID"]) if "Customer_ID" in Xall.columns else Xall
    proba = pipe.predict_proba(Xin)[:,1]
    scored = df.copy()
    scored["Churn_Probability"] = proba
    scored["Risk"] = pd.cut(proba,[0,.35,.65,1],labels=["Low","Medium","High"],include_lowest=True)
    st.dataframe(scored.sort_values("Churn_Probability",ascending=False).head(50),use_container_width=True)
    st.plotly_chart(px.histogram(scored,x="Churn_Probability",color="Risk",nbins=30,title="Risk distribution"),
                    use_container_width=True)
    st.download_button("Download scored customers",scored.to_csv(index=False),"churn_scored.csv","text/csv")
    st.markdown("### Retention playbook")
    st.write("- Prioritize high-risk customers with personalized offers or service recovery.")
    st.write("- Investigate support, delivery, returns and satisfaction signals.")
    st.write("- Test retention actions using controlled experiments rather than assuming correlation is causal.")
