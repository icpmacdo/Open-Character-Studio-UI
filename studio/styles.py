import streamlit as st

def inject_styles() -> None:
    """Add a bit of personality to the Streamlit defaults."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');
        
        /* Global Reset & Typography */
        body {
            background: radial-gradient(circle at 20% 20%, #f8f9fa, #e9ecef 40%, #dee2e6 75%);
            color: #212529;
            font-family: 'IBM Plex Sans', sans-serif;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Space Grotesk', sans-serif;
            color: #0f1c3f;
        }

        /* Hero Section */
        .studio-hero {
            background: linear-gradient(135deg, #0f1c3f 0%, #243b55 100%);
            color: #f8f9fa;
            padding: 32px 40px;
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(15, 28, 63, 0.15);
            margin-bottom: 32px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .studio-hero h1 {
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 600;
            margin-top: 8px;
            margin-bottom: 12px;
        }
        
        .studio-hero p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            max-width: 600px;
        }

        /* Badge */
        .studio-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.15);
            color: #e9ecef;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            backdrop-filter: blur(4px);
        }

        /* Metric Cards */
        .metric-card {
            padding: 20px;
            border-radius: 16px;
            border: 1px solid #e9ecef;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.06);
        }

        /* Buttons */
        .stButton button {
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: all 0.2s ease;
        }
        
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #0f1c3f 0%, #243b55 100%);
            border: none;
            box-shadow: 0 4px 12px rgba(15, 28, 63, 0.2);
        }
        
        .stButton button[kind="primary"]:hover {
            box-shadow: 0 6px 16px rgba(15, 28, 63, 0.3);
            transform: translateY(-1px);
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            border-radius: 8px;
            border-color: #dee2e6;
        }
        
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #243b55;
            box-shadow: 0 0 0 2px rgba(36, 59, 85, 0.1);
        }

        /* Status Indicators */
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-good { background-color: #2ecc71; }
        .status-bad { background-color: #e74c3c; }
        .status-warn { background-color: #f1c40f; }
        
        </style>
        """,
        unsafe_allow_html=True,
    )
