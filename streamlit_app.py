# streamlit_app.py
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="📧 PhishDetect", layout="centered")
st.title("📧 PhishDetect — Phishing Email Detector")

st.markdown("""
Paste the email subject + body below and click **Analyze**.  
The app shows phishing probability, evidence, and risk category.
""")

text = st.text_area("Email text", height=300)

if st.button("Analyze"):
    if not text.strip():
        st.warning("⚠️ Paste email text first.")
    else:
        try:
            resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=20)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                r = resp.json()
                prob = r.get("prob", 0.0)
                category = r.get("category", "N/A")
                label = r.get("label")
                evidence = r.get("evidence", {})

                st.subheader(f"Phishing probability: {prob:.2f}")

                # Color-coded probability + category
                if prob >= 0.7:
                    st.error(f"🚨 High risk — likely phishing ({category})")
                elif prob >= 0.4:
                    st.warning(f"⚠️ Suspicious — be careful ({category})")
                else:
                    st.success(f"✅ Looks legitimate ({category})")

                # Evidence panel
                st.markdown("### Evidence")
                st.write("**Suspicious links found:**")
                for link in evidence.get("links_found", []):
                    st.markdown(f"- 🔗 `{link}`")
                if evidence.get("personalized_greeting"):
                    st.markdown("- 📝 Personalized greeting detected (e.g., 'Dear Tanmay')")
                if evidence.get("urgency_words"):
                    st.markdown(f"- ⚠️ Urgency words: {', '.join(evidence['urgency_words'])}")

        except Exception as e:
            st.error(f"Request failed: {e}")

st.sidebar.header("About")
st.sidebar.write("""
This is an upgraded phishing email detector.  
It combines ML + heuristics to detect risks and explain evidence.  
For production, run API behind HTTPS and integrate domain similarity checks.
""")
