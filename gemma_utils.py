import subprocess

def generate_report(
        prediction,
        confidence,
        consecutive_clips,
        clip_number,
        seconds,
        total_seconds,
        clip_duration,
        mode = "recorded"
):

    if mode == "live":
        analysis_details = f"""

    Analysis Details:
    Operating Mode: Live Camera
    Current Alert Violence Probability: {confidence:.2f}%
    Consecutive Violent Clips: {consecutive_clips}
    Live Session Elapsed Time: {total_seconds:.2f} seconds
    Temporal Confirmation Processing Time: {seconds:.4f} seconds
    Approximate Model Clip Duration: {clip_duration:.2f} seconds
    """
    else:
        analysis_details = f"""
        Analysis Details:
        Total Clips Analysed: {clip_number}
        Video Duration: {seconds:.2f} seconds
        Total Video Duration: {total_seconds:.2f} seconds
        Average Clip Duration: {clip_duration:.2f} seconds
        """    

    prompt = f"""
    You are an Intelligent AI Powered CCTV surveillance system assisting professional security personnel.
    
    Prediction: {prediction}
    Confidence: {confidence:.2f}%
    Consecutive Violent Clips: {consecutive_clips}

    {analysis_details}

    Based on the above information, generate a concise and informative report for the security team.
    The report should be written in the style of an incident report procuced by an intelligent CCTV surveillance system for security personnel.
    Generate a consice security report inclding the following information:

    - Threat Level: Based on the prediction and confidence, assess the threat level (e.g., Low, Medium, High).
    - Explanation: Provide a brief explanation of the prediction and its implications for security.
    - Recommended Actions: Suggest appropriate actions for the security team to take in response to the prediction.
    - Confidence Assessment: Comment on the confidence level of the prediction and any factors that may affect its reliability.

    When operating in Live Camera mode:
    - "Live Session Elapsed Time" is not a video duration.
    - "Temporal Confirmation Processing Time" is computational wall-clock time,
    - Do not state that consecutiveclips "total" the processing time.
    - Do not infer an event

    Additional Instructions:
    Base the report strictly on the provided information. Do not invent additional details or make assumptions beyond the given data. The report should be clear, professional, and actionable for security personnel.
    Generate a formal security report only.
    Do not ask follow-up questions.
    Do not offer further assistance.
    Do not infer or mention an overall average violence probability unless one is explicitly provided.
    Do not describe a probability below 50% as high confidence or high probability.
    Do not describe the supplied video duration as an averae duration.
    When the temporal rule triggers, explain that the alert was caused by sustained consecutive violent predictions.
    Do not create an incident ID, date, time, location, names, or any metadata. Do not include any additional information beyond the specified report content.
    Do not describe confidence as correlation, certainty, proof, or any other term. Use only the term "confidence" to describe the reliability of the prediction.
    For high threat assessments, recommend immediate security measures, such as alerting security personnel, initiating.
    Do not refer to information as "observed", "detected", or "identified". Use neutral language to describe the prediction and its implications unless it is direct.
    For low threat assessments, do not recommend deploying additional personnel or emergency response unless explicitly supported by supplied information.
    Omit unavaiable fields entierly.
    Limit the report to a maximum of 200 words.
    End the report with a clear and concise summary of the security assessment and recommended actions.
    End immediately after providing the security report.
    """

    result = subprocess.run(
        ["ollama", "run", "gemma3"],
        input = prompt,
        text = True,
        capture_output = True,
        timeout = 120,
    )

    if result.returncode != 0:
        return f"Gemma error: {result.stderr.strip()}"

    return result.stdout.strip()