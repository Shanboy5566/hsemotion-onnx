def sadness_normalization(emotion_scores, sadness_id=6, offset=0.0):
    emotion_scores[sadness_id] += offset
    return emotion_scores

def emotion_to_branch(emotion):
    if emotion in ['Anger', 'Disgust', 'Fear', 'Sadness']:
        return 'Negative'
    elif emotion in ['Happiness', 'Surprise']:
        return 'Positive'
    else:
        return 'Neutral'