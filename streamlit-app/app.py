import logging
from typing import Any

import pickle
import streamlit as st
import streamlit.components.v1 as components

import helper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _inject_meta() -> None:
    """Inject basic meta tags and JSON-LD to improve SEO/social previews.

    Streamlit doesn't allow modifying the <head> directly; injecting a small
    HTML block in the body helps some crawlers and social bots pick up Open
    Graph and JSON-LD metadata when supported by the hosting platform.
    """
    meta_html = """
    <meta name="description" content="Detect duplicate question pairs using a trained model. Enter two questions to check if they're duplicates." />
    <meta property="og:title" content="Duplicate Question Pair Detector" />
    <meta property="og:description" content="Enter two questions and the model predicts if they are duplicates." />
    <meta property="og:type" content="website" />
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "WebApplication", "name": "Duplicate Question Pair Detector", "description": "Detect duplicate question pairs using a trained ML model."}
    </script>
    """
    # components.html places content in the body; that's acceptable for many
    # crawlers and for embedding structured data. Keep it non-interactive.
    components.html(meta_html, height=0)


@st.cache_resource
def load_model(path: str = "model.pkl") -> Any:
    """Load the pickled prediction model once and cache it for the session."""
    logger.info("Loading model from %s", path)
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    st.set_page_config(
        page_title="Duplicate Question Pair Detector",
        page_icon="🔎",
        layout="centered",
    )

    _inject_meta()

    st.title("Duplicate Question Pair Detector")
    st.caption("Enter two questions and the model will predict whether they are duplicates.")

    q1 = st.text_input("Question 1", placeholder="Type the first question here")
    q2 = st.text_input("Question 2", placeholder="Type the second question here")

    model = load_model()

    if st.button("Check Pair"):
        if not q1 or not q2:
            st.warning("Please provide both questions before checking.")
            return

        try:
            query = helper.query_point_creator(q1, q2)
            prediction = model.predict(query)[0]
        except Exception as exc:
            logger.exception("Prediction failed")
            st.error("An error occurred while making the prediction.")
            return

        if int(prediction) == 1:
            st.success("Result: Duplicate question pair")
        else:
            st.info("Result: Not a duplicate")

    st.markdown("---")
    st.markdown(
        "Built with a trained ML model — see the project README for details."
    )


if __name__ == "__main__":
    main()


