def sadness_normalization(emotion_scores, sadness_id=6, offset=0.0):
    emotion_scores[sadness_id] += offset
    return emotion_scores

def emotion_to_branch(emotion):
    if emotion in ['Anger', 'Disgust', 'Fear', 'Sadness']:
        return 'Negative', (0, 0, 255)
    elif emotion in ['Happiness', 'Surprise', 'Contempt']:
        return 'Positive', (0, 255, 0)
    else:
        return 'Neutral', (0, 255, 255)