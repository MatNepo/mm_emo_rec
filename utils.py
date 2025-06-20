def transform_results(results):
    """
    Transform results into the required format by:
    1. Removing top_emotion from emotions dictionary in ted section
    2. Adding top_emotion field outside emotions dictionary with the text value of the emotion that had the highest value
    3. Removing unnecessary fields from FED results
    4. Calculating combined_top_emotions
    """
    if not results:
        return results
        
    # Handle TED results
    if results.get('ted') is not None and isinstance(results['ted'], dict):
        ted_emotions = results['ted'].get('emotions')
        if ted_emotions and isinstance(ted_emotions, dict):
            # Remove top_emotion if it exists
            if 'top_emotion' in ted_emotions:
                del ted_emotions['top_emotion']
            
            # Find the emotion with the highest value
            if ted_emotions:
                top_emotion = max(ted_emotions.items(), key=lambda x: x[1])[0]
                results['ted']['top_emotion'] = top_emotion
    
    # Handle FED results - remove unnecessary fields
    if results.get('fed') is not None and isinstance(results['fed'], dict):
        fields_to_remove = ['total_frames', 'frames_with_faces', 'total_faces_detected', 'average_processing_time_ms']
        for field in fields_to_remove:
            results['fed'].pop(field, None)
            
        # Ensure FED emotions sum to 1.0 (100%)
        fed_emotions = results['fed'].get('emotions')
        if fed_emotions and isinstance(fed_emotions, dict):
            total_fed_emotions = sum(fed_emotions.values())
            normalized_fed_emotions = {}
            if total_fed_emotions > 0:
                for emotion, count in fed_emotions.items():
                    normalized_fed_emotions[emotion] = count / total_fed_emotions
                
                # Adjust for potential floating point inaccuracies to ensure sum is exactly 1.0
                current_sum = sum(normalized_fed_emotions.values())
                if current_sum != 1.0 and normalized_fed_emotions:
                    # Find emotion with max value to adjust
                    max_emotion = max(normalized_fed_emotions, key=normalized_fed_emotions.get)
                    normalized_fed_emotions[max_emotion] += (1.0 - current_sum)
            else:
                # If no FED emotions detected or total is 0, set all to 0.0
                for emotion in fed_emotions.keys():
                    normalized_fed_emotions[emotion] = 0.0
            
            results['fed']['emotions'] = normalized_fed_emotions
    
    # Calculate combined_top_emotions
    combined = {}
    
    # Define audio emotions that indicate happiness/joy
    joy_indicators = {
        'Laughter': 0.3,  # Increased weight for laughter
        'Snicker': 0.2,   # Increased weight for snicker
        'Chuckle, chortle': 0.2,  # Increased weight for chuckle
        'Giggle': 0.15,   # Increased weight for giggle
        'Singing': 0.1,   # Some weight for singing
        'Male singing': 0.1  # Some weight for male singing
    }
    
    # Process text emotions (TED)
    if results.get('ted') is not None and isinstance(results['ted'], dict):
        ted_emotions = results['ted'].get('emotions', {})
        if ted_emotions and isinstance(ted_emotions, dict):
            for emotion, value in ted_emotions.items():
                if emotion != 'top_emotion':  # Skip the top_emotion field
                    combined[emotion] = value * 0.4  # Base weight for text emotions
    
    # Process facial emotions (FED)
    if results.get('fed') is not None and isinstance(results['fed'], dict):
        fed_emotions = results['fed'].get('emotions', {})
        if fed_emotions and isinstance(fed_emotions, dict):
            for emotion, value in fed_emotions.items():
                if emotion in combined:
                    combined[emotion] += value * 0.4  # Base weight for facial emotions
                else:
                    combined[emotion] = value * 0.4
    
    # Process audio emotions (AED) and add bonus for joy indicators
    if results.get('aed') is not None and isinstance(results['aed'], dict):
        aed_emotions = results['aed'].get('emotions', {})
        if aed_emotions and isinstance(aed_emotions, dict):
            # Check for joy indicators in top emotions
            for emotion, value in aed_emotions.items():
                if emotion in joy_indicators:
                    # Add bonus to радость based on the joy indicator's weight
                    if 'радость' in combined:
                        combined['радость'] += value * joy_indicators[emotion]
                    else:
                        combined['радость'] = value * joy_indicators[emotion]
    
    # Normalize the combined emotions
    total = sum(combined.values())
    if total > 0:
        combined = {k: v/total for k, v in combined.items()}
    
    # Sort by value in descending order and get top 3
    results['combined_top_emotions'] = dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3])
    
    return results 