# utils.py
def classify_emotion(percentages):
    """
    percentages = dict with keys:
    happy, sad, angry, fear, neutral, surprise, disgust
    """

    # Rule 1: Strong negative
    if (percentages["sad"] > 40 or 
        percentages["angry"] > 40 or 
        percentages["fear"] > 40):
        return "Negative"

    # Rule 2: Strong positive
    if percentages["happy"] > 40:
        return "Positive"

    # Rule 3: Neutral dominance
    if percentages["neutral"] > 40 and percentages["happy"] < 30:
        return "Little Positive"

    # Fallback: Negative
    return "Negative"
